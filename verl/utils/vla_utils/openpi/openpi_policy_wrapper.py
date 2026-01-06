"""
OpenPI Policy Wrapper for SimpleVLA-RL

This module provides a direct wrapper for OpenPI models that uses the raw PyTorch model
for RL training, avoiding the complex policy inference system designed for deployment.

Location: verl/utils/vla_utils/openpi/
"""

import logging
import os
import sys
import pathlib
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn

# Add openpi to path if needed
openpi_path = "/data/zhengshenli/leo/embodied_ai/openpi/src"
if openpi_path not in sys.path:
    sys.path.insert(0, openpi_path)

# Try to import OpenPI modules safely
OPENPI_IMPORT_ERROR = None
_openpi_modules = None

def _try_import_openpi():
    """Lazy import OpenPI modules to avoid orbax/tensorstore conflicts"""
    global OPENPI_IMPORT_ERROR, _openpi_modules
    if _openpi_modules is not None:
        return _openpi_modules
    
    try:
        # Import only what's needed for raw model access
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models import model as openpi_model
        
        _openpi_modules = {
            'pi0_pytorch': PI0Pytorch,
            'pi0_config': Pi0Config,
            'model': openpi_model,
        }
        return _openpi_modules
    except ImportError as e:
        OPENPI_IMPORT_ERROR = e
        return None

logger = logging.getLogger(__name__)


class OpenPIPolicyWrapper(nn.Module):
    """
    Direct wrapper for OpenPI models for RL training.
    
    This wrapper uses the raw PyTorch OpenPI model directly, avoiding the complex
    policy inference system. It's designed for RL training where we need to:
    1. Call model.sample_actions() directly
    2. Handle data format conversion ourselves
    3. Avoid the KV cache and complex inference pipeline
    """
    
    def __init__(
        self,
        model_path: str,
        config_name: str = "pi05_libero",
        action_dim: int = 7,
        action_horizon: int = 8,
        device: str = "cuda",
    ):
        """
        Initialize OpenPI policy wrapper.
        
        Args:
            model_path: Path to the pretrained OpenPI model checkpoint
            config_name: Name of the config (e.g., 'pi05_libero', 'pi0_libero')
            action_dim: Dimension of the action space
            action_horizon: Number of action steps to predict
            device: Device to load the model on
        """
        super().__init__()
        
        self.model_path = model_path
        self.config_name = config_name
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = device
        
        # Lazy import OpenPI modules
        modules = _try_import_openpi()
        if modules is None:
            raise RuntimeError(f"Failed to import OpenPI: {OPENPI_IMPORT_ERROR}")
        
        PI0Pytorch = modules['pi0_pytorch']
        Pi0Config = modules['pi0_config']
        openpi_model_module = modules['model']
        
        # Determine if pi05 based on config name
        pi05 = "pi05" in config_name or "pi_05" in config_name
        
        # Create config
        self.config = Pi0Config(
            pi05=pi05,
            action_dim=action_dim,
            action_horizon=action_horizon,
            max_token_len=200 if pi05 else 48,
            dtype="bfloat16",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m"
        )
        
        # Load model
        logger.info(f"Loading OpenPI model from {model_path}")
        self.model = PI0Pytorch(self.config)
        
        # Load checkpoint
        self._load_checkpoint(model_path)
        
        # Load normalization stats
        self.norm_stats = self._load_norm_stats()
        
        # Move to device
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"OpenPI policy model loaded successfully")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        try:
            import safetensors.torch
            
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_path):
                safetensors.torch.load_model(self.model, model_path)
                logger.info(f"Loaded model weights from {model_path}")
            else:
                logger.warning(f"No checkpoint found at {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    def _load_norm_stats(self) -> Dict[str, Any]:
        """Load normalization statistics from checkpoint."""
        try:
            # Try to load from assets directory
            assets_path = pathlib.Path(self.model_path) / "assets"
            if assets_path.exists():
                import glob
                # Look for norm_stats files
                norm_files = glob.glob(str(assets_path / "**/norm_stats.*"), recursive=True)
                if norm_files:
                    try:
                        import openpi.shared.normalize as _normalize
                        norm_stats = _normalize.load(str(assets_path))
                        logger.info(f"Loaded norm stats from {assets_path}")
                        return norm_stats
                    except Exception as e:
                        logger.warning(f"Failed to load norm stats via OpenPI normalize: {e}")
            
            # Try to load from JSON if available
            import json
            norm_json_path = pathlib.Path(self.model_path) / "norm_stats.json"
            if norm_json_path.exists():
                with open(norm_json_path, 'r') as f:
                    norm_stats = json.load(f)
                logger.info("Loaded norm stats from norm_stats.json")
                return norm_stats
            
            logger.warning("No norm_stats found, using default (identity normalization)")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load norm stats: {e}")
            return {}
    
    def _create_observation(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_description: Optional[Union[str, list]] = None,
    ) -> Any:
        """
        Create OpenPI Observation object from SimpleVLA format.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W] 
            state: Proprioceptive state [B, state_dim]
            input_ids: Tokenized text (for compatibility)
            attention_mask: Attention mask (for compatibility)
            task_description: Task description strings (for π₀.₅ models)
        
        Returns:
            OpenPI Observation object
        """
        batch_size = pixel_values.shape[0]
        
        # Handle multiple images
        if pixel_values.ndim == 5:
            # [B, N_imgs, C, H, W] -> dict of images
            images = {
                f'image_{i}': pixel_values[:, i]
                for i in range(pixel_values.shape[1])
            }
            image_masks = {
                f'image_{i}': torch.ones(batch_size, dtype=torch.bool, device=pixel_values.device)
                for i in range(pixel_values.shape[1])
            }
        else:
            # Single image
            images = {'base_0_rgb': pixel_values}
            image_masks = {'base_0_rgb': torch.ones(batch_size, dtype=torch.BoolTensor).to(pixel_values.device)}
        
        # Handle wrist images
        if wrist_pixel_values is not None:
            if wrist_pixel_values.ndim == 5:
                # Multiple wrist images
                for i in range(wrist_pixel_values.shape[1]):
                    images[f'left_wrist_{i}_rgb'] = wrist_pixel_values[:, i]
                    image_masks[f'left_wrist_{i}_rgb'] = torch.ones(batch_size, dtype=torch.BoolTensor).to(pixel_values.device)
            else:
                # Single wrist image
                images['left_wrist_0_rgb'] = wrist_pixel_values
                image_masks['left_wrist_0_rgb'] = torch.ones(batch_size, dtype=torch.BoolTensor).to(pixel_values.device)
        
        # Default state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.action_dim, device=pixel_values.device)
        
        # Prepare prompt for π₀.₅ models
        if task_description is not None:
            if isinstance(task_description, str):
                prompts = [task_description] * batch_size
            elif isinstance(task_description, list):
                prompts = task_description
            else:
                prompts = ["complete the task"] * batch_size
        else:
            prompts = ["complete the task"] * batch_size
        
        # Create OpenPI Observation object
        try:
            # Try to use official Observation class
            from openpi.models import model as openpi_model
            observation = openpi_model.Observation(
                images=images,
                image_masks=image_masks,
                state=state,
                tokenized_prompt=input_ids if input_ids is not None else torch.zeros(batch_size, 1, dtype=torch.long, device=pixel_values.device),
                tokenized_prompt_mask=attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
                token_ar_mask=attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
                token_loss_mask=attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
            )
        except:
            # Fallback to simple dict if Observation class fails
            observation = {
                'images': images,
                'image_masks': image_masks,
                'state': state,
                'tokenized_prompt': input_ids if input_ids is not None else torch.zeros(batch_size, 1, dtype=torch.long, device=pixel_values.device),
                'tokenized_prompt_mask': attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
                'token_ar_mask': attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
                'token_loss_mask': attention_mask.bool() if attention_mask is not None else torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(pixel_values.device),
            }
        
        return observation
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_description: Optional[Union[str, list]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using raw OpenPI model.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            input_ids: Tokenized text (for compatibility)
            attention_mask: Attention mask (for compatibility) 
            task_description: Task description strings (for π₀.₅ models)
        
        Returns:
            Dictionary with continuous actions
        """
        # Create observation
        observation = self._create_observation(
            pixel_values=pixel_values,
            wrist_pixel_values=wrist_pixel_values,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_description=task_description,
        )
        
        # Sample actions using raw model
        with torch.no_grad():
            continuous_actions = self.model.sample_actions(
                device=pixel_values.device,
                observation=observation,
                num_steps=10  # Standard inference steps
            )
        
        # Ensure correct shape: [B, action_horizon, action_dim]
        if continuous_actions.dim() == 2:
            continuous_actions = continuous_actions.unsqueeze(0)  # Add batch dimension if missing
        
        return {
            'continuous_actions': continuous_actions,
            'actions': continuous_actions  # Alias for compatibility
        }
    
    def sample_actions(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_description: Optional[Union[str, list]] = None,
        deterministic: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample actions from the policy.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            input_ids: Tokenized text (for compatibility)
            attention_mask: Attention mask (for compatibility)
            task_description: Task description strings (for π₀.₅ models)
            deterministic: Whether to use deterministic sampling
        
        Returns:
            Continuous actions [B, action_horizon, action_dim]
        """
        outputs = self.forward(
            pixel_values=pixel_values,
            wrist_pixel_values=wrist_pixel_values,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_description=task_description,
            **kwargs
        )
        
        return outputs['continuous_actions']
    
    def _unnormalize_actions(self, actions: torch.Tensor, unnorm_key: str) -> torch.Tensor:
        """Unnormalize actions using stored statistics"""
        if unnorm_key not in self.norm_stats:
            return actions
        
        stats = self.norm_stats[unnorm_key]
        if 'action' not in stats:
            return actions
        
        action_stats = stats['action']
        
        # Convert from normalized [-1, 1] to original scale
        mean = torch.tensor(action_stats['mean'], device=actions.device)
        std = torch.tensor(action_stats['std'], device=actions.device)
        
        actions_unnorm = actions * std + mean
        
        return actions_unnorm
    
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_description: Optional[Union[str, list]] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions with optional unnormalization.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioceptive state [B, state_dim]
            input_ids: Tokenized text (for compatibility)
            attention_mask: Attention mask (for compatibility)
            task_description: Task description strings (for π₀.₅ models)
            unnorm_key: Key for unnormalization stats
        
        Returns:
            Unnormalized continuous actions [B, action_horizon, action_dim]
        """
        # Get normalized actions
        actions = self.sample_actions(
            pixel_values=pixel_values,
            wrist_pixel_values=wrist_pixel_values,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_description=task_description,
            **kwargs
        )
        
        # Unnormalize if requested
        if unnorm_key is not None and self.norm_stats:
            actions = self._unnormalize_actions(actions, unnorm_key)
        
        return actions


def create_openpi_policy_model(
    model_path: str,
    config_name: str = "pi05_libero",
    action_dim: int = 7,
    action_horizon: int = 8,
    device: str = "cuda",
) -> OpenPIPolicyWrapper:
    """
    Factory function to create an OpenPI policy model for RL training.
    
    Args:
        model_path: Path to the pretrained OpenPI model checkpoint
        config_name: Name of the config (e.g., 'pi05_libero', 'pi0_libero')
        action_dim: Dimension of the action space
        action_horizon: Number of action steps to predict
        device: Device to load the model on
    
    Returns:
        OpenPIPolicyWrapper instance
    """
    return OpenPIPolicyWrapper(
        model_path=model_path,
        config_name=config_name,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device,
    )