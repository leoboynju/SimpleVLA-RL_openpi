"""
OpenPI PPO Wrapper - Makes OpenPI models compatible with SimpleVLA PPO training

This wrapper adapts OpenPI's output interface to work with SimpleVLA's
PPO training framework, handling continuous action spaces properly.

Location: verl/utils/vla_utils/openpi/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class OpenPIPPOWrapper(nn.Module):
    """
    Wraps OpenPI model to make it compatible with SimpleVLA PPO training.
    
    Key features:
    - Returns dictionary with 'logits' key for PPO compatibility
    - Computes proper log probabilities for continuous actions
    - Handles training/inference mode switching
    - Manages normalization/denormalization
    - Provides Gaussian distribution for continuous action space
    
    Location: verl/utils/vla_utils/openpi/
    """
    
    def __init__(
        self,
        openpi_model: nn.Module,
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
            openpi_model: The underlying OpenPI model
            action_dim: Dimension of each action
            action_horizon: Number of actions to predict
            action_std_init: Initial standard deviation for Gaussian policy
            learn_std: Whether to learn the standard deviation
            std_min: Minimum allowed std (for stability)
            std_max: Maximum allowed std (for stability)
            use_tanh: Whether to use tanh squashing for bounded actions
        """
        super().__init__()
        self.openpi_model = openpi_model
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
        
        # Store reference to action tokenizer if available (for PI0RLWrapper)
        self.action_tokenizer = None
        if hasattr(openpi_model, 'action_tokenizer'):
            self.action_tokenizer = openpi_model.action_tokenizer
        
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OpenPI model with PPO-compatible interface.
        
        Key difference from OpenVLA: OpenPI uses raw task description strings,
        not tokenized input_ids. The model handles its own text encoding.
        
        Args:
            pixel_values: Main camera images [B, C, H, W]
            wrist_pixel_values: Wrist camera images [B, C, H, W]
            state: Proprioception state [B, state_dim]
            task_descriptions: List of task description strings (NOT tokenized!)
            input_ids: Token IDs (for compatibility, not used by OpenPI)
            attention_mask: Attention mask (for compatibility, not used by OpenPI)
            return_dict: Whether to return dict (always True for PPO)
        
        Returns:
            Dictionary with keys:
                - 'logits': Predicted action means [B, action_horizon * action_dim]
                - 'action_means': Action means [B, action_horizon, action_dim]
                - 'action_stds': Action stds [B, action_horizon, action_dim]
                - 'hidden_states': Hidden states (if available)
        """
        batch_size = pixel_values.shape[0]
        
        # Prepare task description for OpenPI model
        # OpenPI expects task_description as a string or list of strings
        if task_descriptions is None:
            # Use a default task description
            task_descriptions = ["complete the task"] * batch_size
        
        # Call underlying OpenPI model with proper task description format
        with torch.set_grad_enabled(self.training):
            output = self.openpi_model(
                pixel_values=pixel_values,
                wrist_pixel_values=wrist_pixel_values,
                state=state,
                task_description=task_descriptions,  # Key: raw strings, not tokenized!
            )
        
        # Handle different output formats from OpenPI
        if isinstance(output, tuple):
            # OpenPI returns tuple: ([prefix_output, suffix_output], past_key_values)
            hidden_states, _ = output
            # Assume last hidden state contains action predictions
            if isinstance(hidden_states, list):
                action_hidden = hidden_states[-1]  # Take suffix output
            else:
                action_hidden = hidden_states
        elif isinstance(output, dict):
            # Model returns a dictionary (e.g., from PI0RLWrapper)
            action_hidden = output
        else:
            # Assume direct tensor output
            action_hidden = output
        
        # Extract or create action logits
        if isinstance(action_hidden, dict):
            # Dictionary output - extract appropriate keys
            if 'logits' in action_hidden:
                action_logits = action_hidden['logits']
                # Check if these are token logits from PI0RLWrapper (3D: B, H*D, num_bins)
                if len(action_logits.shape) == 3 and self.action_tokenizer is not None:
                    # Convert token logits to continuous actions using soft detokenization
                    continuous_actions = self.action_tokenizer.soft_detokenize(action_logits)
                    # Reshape to [B, action_horizon, action_dim]
                    action_means = continuous_actions.view(batch_size, self.action_horizon, self.action_dim)
                else:
                    # These are already continuous action logits
                    action_means = action_logits.view(batch_size, self.action_horizon, self.action_dim)
            elif 'continuous_actions' in action_hidden:
                action_means = action_hidden['continuous_actions']
                # Ensure correct shape
                if action_means.shape != (batch_size, self.action_horizon, self.action_dim):
                    action_means = action_means.view(batch_size, self.action_horizon, self.action_dim)
            elif 'action_means' in action_hidden:
                action_means = action_hidden['action_means']
            else:
                raise ValueError(f"Dictionary output doesn't contain expected keys. Keys: {list(action_hidden.keys())}")
        elif hasattr(action_hidden, 'logits'):
            action_logits = action_hidden.logits
            # Check if these are token logits from PI0RLWrapper (3D: B, H*D, num_bins)
            if len(action_logits.shape) == 3 and self.action_tokenizer is not None:
                # Convert token logits to continuous actions using soft detokenization
                continuous_actions = self.action_tokenizer.soft_detokenize(action_logits)
                # Reshape to [B, action_horizon, action_dim]
                action_means = continuous_actions.view(batch_size, self.action_horizon, self.action_dim)
            else:
                # These are already continuous action logits
                action_means = action_logits.view(batch_size, self.action_horizon, self.action_dim)
        elif hasattr(action_hidden, 'last_hidden_state'):
            # Project hidden state to action space
            action_logits = action_hidden.last_hidden_state
            action_means = action_logits.view(batch_size, self.action_horizon, self.action_dim)
        else:
            # Assume hidden_state IS the action representation
            action_means = action_hidden.view(batch_size, self.action_horizon, self.action_dim)
        
        # Get std (broadcast across horizon)
        action_stds = self.std.unsqueeze(0).unsqueeze(0).expand_as(action_means)
        
        # Flatten logits for SimpleVLA compatibility
        logits = action_means.reshape(batch_size, -1)  # [B, action_horizon * action_dim]
        
        # Prepare output dict
        result = {
            'logits': logits,  # [B, action_horizon * action_dim]
            'action_means': action_means,  # [B, action_horizon, action_dim]
            'action_stds': action_stds,  # [B, action_horizon, action_dim]
        }
        
        # Include hidden states if available
        if hasattr(action_hidden, 'hidden_states'):
            result['hidden_states'] = action_hidden.hidden_states
        elif isinstance(action_hidden, list):
            result['hidden_states'] = action_hidden
        elif isinstance(action_hidden, dict) and 'hidden_states' in action_hidden:
            result['hidden_states'] = action_hidden['hidden_states']
        
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
            stds: Predicted stds [B, action_horizon, action_dim]
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
    openpi_model: nn.Module,
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
        openpi_model: The underlying OpenPI model
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