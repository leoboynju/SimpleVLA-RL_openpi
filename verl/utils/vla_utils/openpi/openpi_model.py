"""
OpenPI Model Wrapper for SimpleVLA-RL

This module provides a wrapper around OpenPI models (π₀, π₀.₅) to make them
compatible with the SimpleVLA-RL training framework.
"""

import logging
import os
import sys
import pathlib
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

# Add openpi to path if needed
openpi_path = "/data/zhengshenli/leo/embodied_ai/openpi/src"
if openpi_path not in sys.path:
    sys.path.insert(0, openpi_path)

# NOTE: We delay OpenPI imports to avoid orbax/tensorstore conflicts
# The JAX-based OpenPI wrapper is deprecated; use PI0RLWrapper instead
OPENPI_IMPORT_ERROR = None
_openpi_modules = None

def _try_import_openpi():
    """Lazy import OpenPI modules to avoid orbax/tensorstore conflicts"""
    global OPENPI_IMPORT_ERROR, _openpi_modules
    if _openpi_modules is not None:
        return _openpi_modules
    
    try:
        # Import only what's needed, avoid openpi.shared.download
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.models import model as openpi_model
        
        _openpi_modules = {
            'config': openpi_config,
            'policy_config': policy_config,
            'model': openpi_model,
        }
        return _openpi_modules
    except ImportError as e:
        OPENPI_IMPORT_ERROR = e
        return None

logger = logging.getLogger(__name__)


class OpenPIModel(nn.Module):
    """
    Wrapper for OpenPI models to integrate with SimpleVLA-RL framework.
    
    This class wraps π₀ or π₀.₅ models and provides interfaces compatible
    with the SimpleVLA-RL training and rollout system.
    """
    
    def __init__(
        self,
        model_path: str,
        config_name: str = "pi05_libero",
        action_dim: int = 7,
        action_horizon: int = 1,
        use_pytorch: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize OpenPI model wrapper.
        
        Args:
            model_path: Path to the pretrained OpenPI model checkpoint
            config_name: Name of the config (e.g., 'pi05_libero', 'pi0_libero')
            action_dim: Dimension of the action space
            action_horizon: Number of action steps to predict
            use_pytorch: Whether to use PyTorch version (True) or JAX (False)
            device: Device to load the model on
        """
        super().__init__()
        
        self.model_path = model_path
        self.config_name = config_name
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_pytorch = use_pytorch
        self.device = device
        
        # Lazy import OpenPI modules
        modules = _try_import_openpi()
        if modules is None:
            raise RuntimeError(f"Failed to import OpenPI: {OPENPI_IMPORT_ERROR}")
        
        openpi_config = modules['config']
        policy_config = modules['policy_config']
        openpi_model_module = modules['model']
        
        # Load configuration
        logger.info(f"Loading OpenPI config: {config_name}")
        self.train_config = openpi_config.get_config(config_name)
        
        # Check if checkpoint path exists (avoid download which triggers orbax)
        model_path_obj = pathlib.Path(model_path)
        if model_path_obj.exists():
            checkpoint_dir = str(model_path_obj)
        else:
            # Try to find checkpoint in common locations
            raise FileNotFoundError(
                f"OpenPI checkpoint not found at {model_path}. "
                "Please ensure the model path exists (downloading via OpenPI is disabled due to JAX conflicts)."
            )
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        
        # Check if PyTorch model
        weight_path = self.checkpoint_dir / "model.safetensors"
        self.is_pytorch = weight_path.exists() if use_pytorch else False
        
        # Load policy
        logger.info(f"Loading OpenPI policy from {checkpoint_dir}")
        self.policy = policy_config.create_trained_policy(
            self.train_config,
            checkpoint_dir,
            pytorch_device=device if self.is_pytorch else None,
        )
        
        # Load norm stats
        self._load_norm_stats()
        
        logger.info(f"OpenPI model loaded successfully (PyTorch: {self.is_pytorch})")
    
    def _load_norm_stats(self):
        """Load normalization statistics from checkpoint."""
        try:
            # Try to load norm stats directly from assets directory
            assets_path = self.checkpoint_dir / "assets"
            if assets_path.exists():
                import glob
                # Look for norm_stats files
                norm_files = glob.glob(str(assets_path / "**/norm_stats.*"), recursive=True)
                if norm_files:
                    try:
                        import openpi.shared.normalize as _normalize
                        self.norm_stats = _normalize.load(str(assets_path))
                        logger.info(f"Loaded norm stats from {assets_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load norm stats via OpenPI normalize: {e}")
            
            # Try to load from JSON if available
            import json
            norm_json_path = self.checkpoint_dir / "norm_stats.json"
            if norm_json_path.exists():
                with open(norm_json_path, 'r') as f:
                    self.norm_stats = json.load(f)
                logger.info("Loaded norm stats from norm_stats.json")
                return
            
            # Try data config approach (may fail if OpenPI data module has orbax deps)
            try:
                data_config = self.train_config.data.create(
                    self.train_config.assets_dirs,
                    self.train_config.model
                )
                if data_config.asset_id is not None:
                    from openpi.training import checkpoints
                    self.norm_stats = checkpoints.load_norm_stats(
                        self.checkpoint_dir / "assets",
                        data_config.asset_id
                    )
                    logger.info(f"Loaded norm stats for asset: {data_config.asset_id}")
                    return
            except Exception as e:
                logger.debug(f"Could not load norm stats via OpenPI data config: {e}")
            
            self.norm_stats = {}
            logger.warning("No norm_stats found, using default (identity normalization)")
        except Exception as e:
            logger.warning(f"Failed to load norm stats: {e}")
            self.norm_stats = {}
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass compatible with SimpleVLA-RL rollout.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            input_ids: Tokenized language prompt
            attention_mask: Attention mask for prompt
            state: Proprioceptive state [B, state_dim]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            
        Returns:
            Dict with 'logits' containing predicted actions
        """
        batch_size = pixel_values.shape[0]
        
        # Prepare inputs for OpenPI
        inputs = self._prepare_openpi_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            wrist_pixel_values=wrist_pixel_values,
        )
        
        # Run inference
        outputs = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in inputs.items()}
            result = self.policy.infer(example)
            outputs.append(result["actions"])
        
        # Stack outputs
        actions = np.stack(outputs, axis=0)  # [B, action_horizon, action_dim]
        
        # Convert to torch tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(pixel_values.device)
        
        # Flatten to [B, action_horizon * action_dim] for compatibility
        logits = actions.reshape(batch_size, -1)
        
        return {"logits": logits}
    
    def _prepare_openpi_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        wrist_pixel_values: Optional[torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare inputs in the format expected by OpenPI policy.
        
        Args:
            pixel_values: Main camera images [B, C, H, W] or [B, H, W, C]
            input_ids: Tokenized language prompt [B, L]
            attention_mask: Attention mask [B, L]
            state: Proprioceptive state [B, state_dim]
            wrist_pixel_values: Wrist camera images [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Dictionary of inputs for OpenPI
        """
        batch_size = pixel_values.shape[0]
        
        # Convert to numpy and handle channel ordering
        def to_numpy_hwc(img_tensor):
            """Convert tensor to numpy array in HWC format."""
            if isinstance(img_tensor, torch.Tensor):
                img = img_tensor.cpu().numpy()
            else:
                img = img_tensor
            
            # Handle different input formats
            if img.ndim == 4:
                # Batch of images
                if img.shape[1] == 3:  # [B, C, H, W] -> [B, H, W, C]
                    img = np.transpose(img, (0, 2, 3, 1))
            elif img.ndim == 3:
                # Single image
                if img.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                    img = np.transpose(img, (1, 2, 0))
            
            # Convert to uint8 if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            return img
        
        base_images = to_numpy_hwc(pixel_values)
        
        # Prepare wrist images
        if wrist_pixel_values is not None:
            wrist_images = to_numpy_hwc(wrist_pixel_values)
        else:
            # Create dummy wrist images if not provided
            wrist_images = np.zeros_like(base_images)
        
        # Prepare state
        if state is not None:
            if isinstance(state, torch.Tensor):
                state_np = state.cpu().numpy()
            else:
                state_np = state
        else:
            # Default state (8 dimensions for LIBERO)
            state_np = np.zeros((batch_size, 8), dtype=np.float32)
        
        # Prepare prompt (default if not provided)
        # Note: We'll use a default prompt; in practice this should come from the task
        default_prompt = "Complete the task"
        
        # Build inputs dict
        inputs = {
            "observation/state": state_np,
            "observation/image": base_images,
            "observation/wrist_image": wrist_images,
            "prompt": [default_prompt] * batch_size,
        }
        
        return inputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = None,
        **kwargs
    ):
        """
        Generate actions (alias for forward for compatibility).
        """
        return self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            wrist_pixel_values=wrist_pixel_values,
            **kwargs
        )


def create_openpi_model(
    model_path: str,
    config_name: str = "pi05_libero",
    action_dim: int = 7,
    action_horizon: int = 1,
    use_pytorch: bool = True,
    device: str = "cuda",
) -> OpenPIModel:
    """
    Factory function to create an OpenPI model wrapper.
    
    Args:
        model_path: Path to the pretrained OpenPI model checkpoint
        config_name: Name of the config (e.g., 'pi05_libero', 'pi0_libero')
        action_dim: Dimension of the action space
        action_horizon: Number of action steps to predict
        use_pytorch: Whether to use PyTorch version (True) or JAX (False)
        device: Device to load the model on
        
    Returns:
        OpenPIModel instance
    """
    return OpenPIModel(
        model_path=model_path,
        config_name=config_name,
        action_dim=action_dim,
        action_horizon=action_horizon,
        use_pytorch=use_pytorch,
        device=device,
    )

