"""
Compatible OpenPI Wrapper for SimpleVLA-RL

This wrapper provides a unified interface for both directly loaded OpenPI models
and wrapper-based models to ensure compatibility with SimpleVLA-RL's PPO training.

Location: verl/utils/vla_utils/openpi/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import os


class CompatibleOpenPIWrapper(nn.Module):
    """
    Universal wrapper for OpenPI models that provides compatibility with SimpleVLA-RL.
    
    This wrapper can handle:
    1. Directly loaded PI0Pytorch models
    2. Wrapper-based models (PI0RLWrapper)
    3. Different output formats and interfaces
    
    Features:
    - Automatic interface detection and adaptation
    - Proper log probability computation for PPO
    - Action normalization/denormalization support
    - Gradient checkpointing compatibility
    """
    
    def __init__(
        self,
        model: nn.Module,
        action_dim: int = 7,
        action_horizon: int = 8,
        action_std_init: float = 0.1,
        learn_std: bool = True,
        std_min: float = 0.01,
        std_max: float = 1.0,
        use_tanh: bool = False,
        model_type: str = "auto",  # "direct", "wrapper", or "auto"
    ):
        """
        Args:
            model: The underlying OpenPI model (directly loaded or wrapped)
            action_dim: Dimension of each action
            action_horizon: Number of actions to predict
            action_std_init: Initial standard deviation for Gaussian policy
            learn_std: Whether to learn the standard deviation
            std_min: Minimum allowed std (for stability)
            std_max: Maximum allowed std (for stability)
            use_tanh: Whether to use tanh squashing for bounded actions
            model_type: Type of underlying model ("auto", "direct", "wrapper")
        """
        super().__init__()
        self.model = model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_tanh = use_tanh
        self.learn_std = learn_std
        self.std_min = std_min
        self.std_max = std_max
        
        # Detect model type if auto
        if model_type == "auto":
            model_type = self._detect_model_type()
        
        self.model_type = model_type
        
        # Learnable log standard deviation (for exploration)
        if learn_std:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), torch.log(torch.tensor(action_std_init, dtype=torch.float32)))
            )
        else:
            self.register_buffer(
                'log_std',
                torch.full((action_dim,), torch.log(torch.tensor(action_std_init, dtype=torch.float32)))
            )
        
        # Store normalization stats
        self.norm_stats = None
        
        # Track training mode
        self._is_training = True
        
        # Model-specific attributes
        self.num_inference_steps = getattr(model, 'num_inference_steps', 10)
        
    def _detect_model_type(self) -> str:
        """Detect the type of underlying model"""
        # Check for direct PI0Pytorch model
        if hasattr(self.model, 'paligemma_with_expert'):
            return "direct"
        
        # Check for wrapper-based model
        if hasattr(self.model, 'pi0_model'):
            return "wrapper"
        
        # Check for other wrapper types
        if hasattr(self.model, 'action_tokenizer'):
            return "wrapper"
        
        # Default to wrapper for compatibility
        return "wrapper"
    
    @property
    def std(self) -> torch.Tensor:
        """Get current standard deviation with clipping"""
        std = torch.exp(self.log_std)
        return torch.clamp(std, min=self.std_min, max=self.std_max)
    
    def _prepare_observation(self, pixel_values, wrist_pixel_values, state, input_ids, attention_mask, **kwargs):
        """Prepare observation for the underlying model"""
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Handle multiple images
        images = {}
        image_masks = {}
        
        if pixel_values is not None:
            images['base_0_rgb'] = pixel_values
            image_masks['base_0_rgb'] = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        if wrist_pixel_values is not None:
            images['left_wrist_0_rgb'] = wrist_pixel_values
            image_masks['left_wrist_0_rgb'] = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Default state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.action_dim, device=device)
        
        # Get tokenized prompt
        if input_ids is not None:
            tokenized_prompts = input_ids
            prompt_masks = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        else:
            # Create default prompt for compatibility
            tokenized_prompts = torch.zeros(batch_size, 48, dtype=torch.long, device=device)
            prompt_masks = torch.zeros(batch_size, 48, dtype=torch.bool, device=device)
        
        # Try to create observation in the format expected by OpenPI
        try:
            # For direct OpenPI models
            if self.model_type == "direct":
                # Create a simple observation object
                class SimpleObservation:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                observation = SimpleObservation(
                    images=images,
                    image_masks=image_masks,
                    state=state,
                    tokenized_prompt=tokenized_prompts,
                    tokenized_prompt_mask=prompt_masks
                )
                return observation
            
            # For wrapper models
            elif self.model_type == "wrapper":
                if hasattr(self.model, 'prepare_observation'):
                    return self.model.prepare_observation(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        state=state,
                        **kwargs
                    )
                else:
                    # Fallback to simple observation
                    class SimpleObservation:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                setattr(self, k, v)
                    
                    observation = SimpleObservation(
                        images=images,
                        image_masks=image_masks,
                        state=state,
                        tokenized_prompt=tokenized_prompts,
                        tokenized_prompt_mask=prompt_masks
                    )
                    return observation
            
        except Exception as e:
            print(f"Warning: Error preparing observation: {e}")
            # Return basic inputs as fallback
            return {
                'pixel_values': pixel_values,
                'wrist_pixel_values': wrist_pixel_values,
                'state': state,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    def _extract_actions(self, model_output) -> torch.Tensor:
        """Extract actions from model output"""
        try:
            if self.model_type == "direct":
                # Direct OpenPI model returns diffusion-based continuous actions
                if isinstance(model_output, torch.Tensor):
                    actions = model_output
                elif hasattr(model_output, 'continuous_actions'):
                    actions = model_output.continuous_actions
                else:
                    # Assume it's the raw action tensor
                    actions = model_output
                
                # Ensure correct shape
                if actions.dim() == 3:  # [B, H, D]
                    return actions
                elif actions.dim() == 2:  # [B, H*D]
                    return actions.view(-1, self.action_horizon, self.action_dim)
                else:
                    raise ValueError(f"Unexpected action shape: {actions.shape}")
            
            elif self.model_type == "wrapper":
                # Wrapper model returns action logits
                if isinstance(model_output, dict):
                    if 'logits' in model_output:
                        logits = model_output['logits']
                        # Detokenize if necessary
                        if hasattr(self.model, 'action_tokenizer'):
                            actions = self.model.action_tokenizer.soft_detokenize(logits)
                        else:
                            # Assume logits are already in action space
                            actions = logits
                    elif 'continuous_actions' in model_output:
                        actions = model_output['continuous_actions']
                    else:
                        raise ValueError("Wrapper model output doesn't contain expected keys")
                else:
                    # Assume it's already the action tensor
                    actions = model_output
                
                # Ensure correct shape
                if actions.dim() == 3:  # [B, H, D]
                    return actions
                elif actions.dim() == 2:  # [B, H*D]
                    return actions.view(-1, self.action_horizon, self.action_dim)
                else:
                    raise ValueError(f"Unexpected action shape: {actions.shape}")
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
        except Exception as e:
            print(f"Error extracting actions: {e}")
            # Return dummy actions as fallback
            batch_size = 1
            return torch.randn(batch_size, self.action_horizon, self.action_dim)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SimpleVLA-compatible interface.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioception state [B, state_dim]
            input_ids: Token IDs (for compatibility)
            attention_mask: Attention mask (for compatibility)
            return_dict: Whether to return dict (always True for PPO)
        
        Returns:
            Dictionary with:
                - 'logits': Action means [B, action_horizon * action_dim]
                - 'action_means': Action means [B, action_horizon, action_dim]
                - 'action_stds': Action stds [B, action_horizon, action_dim]
        """
        batch_size = pixel_values.shape[0]
        
        # Prepare observation
        observation = self._prepare_observation(
            pixel_values, wrist_pixel_values, state, input_ids, attention_mask, **kwargs
        )
        
        # Call underlying model
        with torch.set_grad_enabled(self.training):
            if self.model_type == "direct":
                # Direct OpenPI model expects observation and returns continuous actions
                model_output = self.model(observation, actions=None)
                if hasattr(model_output, 'sample_actions'):
                    # For inference, use sample_actions
                    continuous_actions = model_output.sample_actions(
                        device=pixel_values.device,
                        observation=observation,
                        num_steps=self.num_inference_steps
                    )
                else:
                    # For training, model_output might be loss or other
                    continuous_actions = model_output
            else:
                # Wrapper model
                if hasattr(self.model, 'forward'):
                    model_output = self.model.forward(
                        observation=observation,
                        return_continuous=True,
                        **kwargs
                    )
                else:
                    model_output = self.model(observation, **kwargs)
                
                continuous_actions = model_output
        
        # Extract actions from model output
        action_means = self._extract_actions(model_output)
        
        # Ensure correct shape: [B, action_horizon, action_dim]
        if action_means.shape != (batch_size, self.action_horizon, self.action_dim):
            action_means = action_means.view(batch_size, self.action_horizon, self.action_dim)
        
        # Get std (broadcast across horizon)
        action_stds = self.std.unsqueeze(0).unsqueeze(0).expand_as(action_means)
        
        # Flatten logits for SimpleVLA compatibility
        logits = action_means.view(batch_size, -1)  # [B, action_horizon * action_dim]
        
        # Prepare output dict
        result = {
            'logits': logits,  # [B, action_horizon * action_dim]
            'action_means': action_means,  # [B, action_horizon, action_dim]
            'action_stds': action_stds,  # [B, action_horizon, action_dim]
        }
        
        return result
    
    def compute_log_prob(
        self,
        actions: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for continuous actions.
        
        Args:
            actions: Actions taken [B, action_horizon, action_dim]
            means: Predicted action means [B, action_horizon, action_dim]
            stds: Standard deviations [B, action_horizon, action_dim]
            action_mask: Mask for valid actions [B, action_horizon]
        
        Returns:
            log_probs: Log probabilities [B, action_horizon]
            entropy: Entropy [B, action_horizon]
        """
        # Create Gaussian distribution
        dist = torch.distributions.Normal(means, stds)
        
        # Compute log probability (sum over action dimensions)
        log_probs = dist.log_prob(actions)  # [B, action_horizon, action_dim]
        log_probs = log_probs.sum(dim=-1)  # [B, action_horizon]
        
        # Compute entropy (sum over action dimensions)
        entropy = dist.entropy()  # [B, action_horizon, action_dim]
        entropy = entropy.sum(dim=-1)  # [B, action_horizon]
        
        # Apply action mask if provided
        if action_mask is not None:
            mask = action_mask.unsqueeze(-1)
            log_probs = log_probs * mask.squeeze(-1)
            entropy = entropy * mask.squeeze(-1)
        
        return log_probs, entropy
    
    def sample_actions(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Sample actions from Gaussian policy.
        
        Args:
            means: Predicted action means [B, action_horizon, action_dim]
            stds: Standard deviations [B, action_horizon, action_dim]
            deterministic: If True, return means without sampling
        
        Returns:
            actions: Sampled actions [B, action_horizon, action_dim]
        """
        if deterministic:
            return means
        
        # Sample from Gaussian
        dist = torch.distributions.Normal(means, stds)
        actions = dist.sample()  # [B, action_horizon, action_dim]
        
        # Apply tanh squashing if enabled
        if self.use_tanh:
            actions = torch.tanh(actions)
        
        return actions
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self._is_training = mode
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self._is_training = False
        self.model.eval()
        return super().eval()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing if supported"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing if supported"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
    
    def get_config(self) -> Dict[str, Any]:
        """Get wrapper configuration"""
        return {
            'action_dim': self.action_dim,
            'action_horizon': self.action_horizon,
            'action_std_init': float(torch.exp(self.log_std[0]).item()),
            'learn_std': self.learn_std,
            'std_min': self.std_min,
            'std_max': self.std_max,
            'use_tanh': self.use_tanh,
            'model_type': self.model_type,
            'training': self._is_training,
            'num_inference_steps': self.num_inference_steps,
        }


def create_compatible_wrapper(
    model: nn.Module,
    action_dim: int = 7,
    action_horizon: int = 8,
    action_std_init: float = 0.1,
    learn_std: bool = True,
    std_min: float = 0.01,
    std_max: float = 1.0,
    use_tanh: bool = False,
    model_type: str = "auto",
) -> CompatibleOpenPIWrapper:
    """
    Factory function to create a compatible OpenPI wrapper.
    
    Args:
        model: The underlying OpenPI model
        action_dim: Dimension of each action
        action_horizon: Number of actions to predict
        action_std_init: Initial standard deviation for Gaussian policy
        learn_std: Whether to learn the standard deviation
        std_min: Minimum allowed std (for stability)
        std_max: Maximum allowed std (for stability)
        use_tanh: Whether to use tanh squashing for bounded actions
        model_type: Type of underlying model ("auto", "direct", "wrapper")
    
    Returns:
        CompatibleOpenPIWrapper instance
    """
    return CompatibleOpenPIWrapper(
        model=model,
        action_dim=action_dim,
        action_horizon=action_horizon,
        action_std_init=action_std_init,
        learn_std=learn_std,
        std_min=std_min,
        std_max=std_max,
        use_tanh=use_tanh,
        model_type=model_type,
    )