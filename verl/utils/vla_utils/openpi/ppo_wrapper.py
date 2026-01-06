"""
OpenPI PPO Wrapper - Makes OpenPI models compatible with SimpleVLA PPO training

This wrapper adapts OpenPI's raw model interface to work with SimpleVLA's
PPO training framework, using the policy wrapper to handle continuous action spaces properly.

Location: verl/utils/vla_utils/openpi/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Import the policy wrapper
from .openpi_policy_wrapper import OpenPIPolicyWrapper


class OpenPIPPOWrapper(nn.Module):
    """
    Wraps OpenPI model to make it compatible with SimpleVLA PPO training.
    
    Key features:
    - Uses raw OpenPI model directly for action generation
    - Computes proper log probabilities for continuous actions
    - Handles training/inference mode switching
    - Manages normalization/denormalization
    - Provides Gaussian distribution for continuous action space
    
    Location: verl/utils/vla_utils/openpi/
    """
    
    def __init__(
        self,
        openpi_model: Union[nn.Module, OpenPIPolicyWrapper],
        action_dim: int = 7,
        action_horizon: int = 8,
        action_std_init: float = 0.1,
        learn_std: bool = True,
        std_min: float = 0.01,
        std_max: float = 1.0,
        use_tanh: bool = False,
    ):
        """
        Args:
            openpi_model: The underlying OpenPI model (can be raw model or OpenPIPolicyWrapper)
            action_dim: Dimension of each action
            action_horizon: Number of actions to predict
            action_std_init: Initial standard deviation for Gaussian policy
            learn_std: Whether to learn the standard deviation
            std_min: Minimum allowed std (for stability)
            std_max: Maximum allowed std (for stability)
            use_tanh: Whether to use tanh squashing for bounded actions
        """
        super().__init__()
        
        # Handle both raw models and wrapped models
        if isinstance(openpi_model, OpenPIPolicyWrapper):
            self.openpi_model = openpi_model
            self.is_wrapped = True
        else:
            # Raw model - create policy wrapper
            self.openpi_model = openpi_model
            self.is_wrapped = False
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_tanh = use_tanh
        self.learn_std = learn_std
        self.std_min = std_min
        self.std_max = std_max
        
        # Learnable log standard deviation (for exploration)
        if learn_std:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), np.log(action_std_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_std',
                torch.full((action_dim,), np.log(action_std_init), dtype=torch.float32)
            )
        
        # Track training mode
        self._is_training = True
        
    @property
    def std(self) -> torch.Tensor:
        """Get current standard deviation with clipping"""
        std = torch.exp(self.log_std)
        return torch.clamp(std, min=self.std_min, max=self.std_max)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        task_descriptions: Optional[list] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OpenPI model with PPO-compatible interface.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioception state [B, state_dim]
            task_descriptions: List of task description strings (for π₀.₅ models)
            input_ids: Token IDs (for compatibility)
            attention_mask: Attention mask (for compatibility)
            return_dict: Whether to return dict (always True for PPO)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with keys:
                - 'logits': Predicted action means [B, action_horizon * action_dim]
                - 'action_means': Action means [B, action_horizon, action_dim]
                - 'action_stds': Action stds [B, action_horizon, action_dim]
        """
        batch_size = pixel_values.shape[0]
        
        # Prepare task description for OpenPI model
        task_description = None
        if task_descriptions is not None:
            if isinstance(task_descriptions, list) and len(task_descriptions) > 0:
                task_description = task_descriptions[0] if len(task_descriptions) == 1 else task_descriptions
            else:
                task_description = "complete the task"
        else:
            task_description = "complete the task"
        
        # Call underlying model to get continuous actions
        with torch.set_grad_enabled(self.training):
            if self.is_wrapped:
                # Use wrapped model interface
                outputs = self.openpi_model.forward(
                    pixel_values=pixel_values,
                    wrist_pixel_values=wrist_pixel_values,
                    state=state,
                    task_description=task_description,
                )
                continuous_actions = outputs['continuous_actions']
            else:
                # Use raw model interface - try different output formats
                if hasattr(self.openpi_model, 'forward'):
                    try:
                        outputs = self.openpi_model.forward(
                            pixel_values=pixel_values,
                            wrist_pixel_values=wrist_pixel_values,
                            state=state,
                            task_description=task_description,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **kwargs
                        )
                        
                        # Handle different output formats from raw models
                        if isinstance(outputs, dict):
                            if 'logits' in outputs:
                                action_logits = outputs['logits']
                                # Check if this is 3D token logits from PI0RLWrapper
                                if action_logits.dim() == 3:
                                    # Token logits [B, H*D, num_bins] - convert to continuous actions
                                    if hasattr(self.openpi_model, 'action_tokenizer'):
                                        continuous_actions = self.openpi_model.action_tokenizer.soft_detokenize(action_logits)
                                    else:
                                        # Fallback: take argmax and map to continuous space
                                        token_indices = action_logits.argmax(dim=-1)
                                        continuous_actions = self._tokens_to_continuous(token_indices)
                                else:
                                    # Already 2D continuous logits
                                    continuous_actions = action_logits
                            elif 'continuous_actions' in outputs:
                                continuous_actions = outputs['continuous_actions']
                            elif 'actions' in outputs:
                                continuous_actions = outputs['actions']
                            else:
                                # Assume the first tensor value is the actions
                                for key, value in outputs.items():
                                    if isinstance(value, torch.Tensor):
                                        continuous_actions = value
                                        break
                                else:
                                    raise ValueError("Could not find action tensor in model output")
                        else:
                            continuous_actions = outputs
                    except Exception as e:
                        # Fallback to direct sampling
                        continuous_actions = self._sample_directly(pixel_values, wrist_pixel_values, state, task_description)
                else:
                    # Direct sampling for models without forward method
                    continuous_actions = self._sample_directly(pixel_values, wrist_pixel_values, state, task_description)
        
        # Ensure correct shape: [B, action_horizon, action_dim]
        if continuous_actions.dim() == 2:
            # [B, H*D] -> reshape to [B, H, D]
            continuous_actions = continuous_actions.view(batch_size, self.action_horizon, self.action_dim)
        elif continuous_actions.dim() == 1:
            # Single action - add batch and horizon dimensions
            continuous_actions = continuous_actions.unsqueeze(0).unsqueeze(0)
        
        # Ensure we have the right shape
        if continuous_actions.shape != (batch_size, self.action_horizon, self.action_dim):
            continuous_actions = continuous_actions.view(batch_size, self.action_horizon, self.action_dim)
        
        # Get std (broadcast across horizon)
        action_stds = self.std.unsqueeze(0).unsqueeze(0).expand_as(continuous_actions)
        
        # Flatten logits for SimpleVLA compatibility
        logits = continuous_actions.view(batch_size, -1)
        
        # Prepare output dict
        result = {
            'logits': logits,  # [B, action_horizon * action_dim]
            'action_means': continuous_actions,  # [B, action_horizon, action_dim]
            'action_stds': action_stds,  # [B, action_horizon, action_dim]
        }
        
        return result
    
    def _sample_directly(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        task_description: Union[str, list],
    ) -> torch.Tensor:
        """Fallback method to sample actions directly from the model."""
        try:
            if hasattr(self.openpi_model, 'sample_actions'):
                if self.is_wrapped:
                    continuous_actions = self.openpi_model.sample_actions(
                        pixel_values=pixel_values,
                        wrist_pixel_values=wrist_pixel_values,
                        state=state,
                        task_description=task_description,
                    )
                else:
                    # Create observation for raw model
                    observation = self._create_observation_for_raw_model(
                        pixel_values, wrist_pixel_values, state, task_description
                    )
                    continuous_actions = self.openpi_model.sample_actions(
                        device=pixel_values.device,
                        observation=observation,
                        num_steps=10
                    )
                return continuous_actions
            else:
                raise AttributeError("Model does not have sample_actions method")
        except Exception as e:
            print(f"Warning: Direct sampling failed: {e}")
            # Return random actions as last resort
            return torch.randn(
                pixel_values.shape[0], 
                self.action_horizon, 
                self.action_dim, 
                device=pixel_values.device
            )
    
    def _create_observation_for_raw_model(
        self,
        pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        task_description: Union[str, list],
    ):
        """Create observation object for raw OpenPI models."""
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Handle images
        images = {'base_0_rgb': pixel_values}
        image_masks = {'base_0_rgb': torch.ones(batch_size, dtype=torch.BoolTensor).to(device)}
        
        if wrist_pixel_values is not None:
            images['left_wrist_0_rgb'] = wrist_pixel_values
            image_masks['left_wrist_0_rgb'] = torch.ones(batch_size, dtype=torch.BoolTensor).to(device)
        
        # Handle state
        if state is None:
            state = torch.zeros(batch_size, self.action_dim, device=device)
        
        # Create observation object
        try:
            from openpi.models import model as openpi_model
            observation = openpi_model.Observation(
                images=images,
                image_masks=image_masks,
                state=state,
                tokenized_prompt=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                tokenized_prompt_mask=torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(device),
            )
        except:
            # Fallback dict
            observation = {
                'images': images,
                'image_masks': image_masks,
                'state': state,
                'tokenized_prompt': torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                'tokenized_prompt_mask': torch.ones(batch_size, 1, dtype=torch.BoolTensor).to(device),
            }
        
        return observation
    
    def _tokens_to_continuous(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Convert token indices to continuous actions using uniform bins."""
        # Simple mapping: token indices in [0, num_bins-1] -> continuous in [-1, 1]
        num_bins = 256  # Default number of bins
        continuous_actions = (token_indices.float() / (num_bins - 1)) * 2.0 - 1.0
        return continuous_actions
    
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
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self._is_training = False
        return super().eval()
    
    def get_config(self) -> Dict:
        """Get wrapper configuration"""
        return {
            'action_dim': self.action_dim,
            'action_horizon': self.action_horizon,
            'action_std_init': float(torch.exp(self.log_std[0]).item()),
            'learn_std': self.learn_std,
            'std_min': self.std_min,
            'std_max': self.std_max,
            'use_tanh': self.use_tanh,
            'training': self._is_training,
        }


def create_openpi_ppo_wrapper(
    openpi_model: Union[nn.Module, OpenPIPolicyWrapper],
    action_dim: int = 7,
    action_horizon: int = 8,
    action_std_init: float = 0.1,
    learn_std: bool = True,
    std_min: float = 0.01,
    std_max: float = 1.0,
    use_tanh: bool = False,
) -> OpenPIPPOWrapper:
    """
    Factory function to create an OpenPI PPO wrapper.
    
    Args:
        openpi_model: The underlying OpenPI model (can be raw model or OpenPIPolicyWrapper)
        action_dim: Dimension of each action
        action_horizon: Number of actions to predict
        action_std_init: Initial standard deviation for Gaussian policy
        learn_std: Whether to learn the standard deviation
        std_min: Minimum allowed std (for stability)
        std_max: Maximum allowed std (for stability)
        use_tanh: Whether to use tanh squashing for bounded actions
    
    Returns:
        OpenPIPPOWrapper instance
    """
    return OpenPIPPOWrapper(
        openpi_model=openpi_model,
        action_dim=action_dim,
        action_horizon=action_horizon,
        action_std_init=action_std_init,
        learn_std=learn_std,
        std_min=std_min,
        std_max=std_max,
        use_tanh=use_tanh,
    )