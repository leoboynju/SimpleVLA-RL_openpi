"""
RL Wrapper for Official OpenPI PyTorch Implementation

This module wraps the official PI0Pytorch model to make it compatible with
SimpleVLA-RL's PPO training framework.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

# Add OpenPI to path
openpi_path = "/data/zhengshenli/leo/embodied_ai/openpi/src"
if openpi_path not in sys.path:
    sys.path.insert(0, openpi_path)

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config

# Import model module safely without JAX dependencies
try:
    from openpi.models import model as openpi_model
except ImportError as e:
    print(f"Warning: Could not import openpi_model: {e}")
    # Create minimal Observation class if import fails
    from collections import namedtuple
    Observation = namedtuple('Observation', ['images', 'image_masks', 'state', 'tokenized_prompt', 'tokenized_prompt_mask'])
    openpi_model = type('Module', (), {'Observation': Observation})()


class ActionTokenizer(nn.Module):
    """
    Convert continuous actions to/from discretized token logits
    
    This enables RL training by discretizing the continuous action space
    into bins that can be treated as classification targets.
    """
    
    def __init__(self, action_dim: int, action_horizon: int, num_bins: int = 256, 
                 temperature: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_bins = num_bins
        
        # Learnable bin centers (initialized to uniform grid in [-1, 1])
        bin_centers = torch.linspace(-1, 1, num_bins)
        self.register_buffer('bin_centers', bin_centers)
        
        # Learnable temperature for soft tokenization
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
    
    @property
    def temperature(self):
        return self.log_temperature.exp()
    
    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous actions to logits over bins
        
        Args:
            actions: [B, H, D] continuous actions in [-1, 1]
        
        Returns:
            logits: [B, H*D, num_bins] logits for each action dimension
        """
        B, H, D = actions.shape
        
        # Reshape to [B*H*D, 1]
        actions_flat = actions.reshape(-1, 1)
        
        # Compute distances to bin centers: [B*H*D, num_bins]
        distances = -torch.abs(actions_flat - self.bin_centers.unsqueeze(0))
        
        # Convert to logits with temperature
        logits = distances / self.temperature
        
        # Reshape to [B, H*D, num_bins]
        logits = logits.reshape(B, H * D, self.num_bins)
        
        return logits
    
    def detokenize(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert action tokens back to continuous actions
        
        Args:
            action_tokens: [B, H*D] token indices
        
        Returns:
            actions: [B, H, D] continuous actions
        """
        # Map token indices to bin centers
        actions_flat = self.bin_centers[action_tokens]  # [B, H*D]
        
        # Reshape to [B, H, D]
        B = actions_flat.shape[0]
        actions = actions_flat.reshape(B, self.action_horizon, self.action_dim)
        
        return actions
    
    def soft_detokenize(self, action_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to continuous actions using soft (weighted) mapping
        
        Args:
            action_logits: [B, H*D, num_bins] logits
        
        Returns:
            actions: [B, H, D] continuous actions
        """
        # Compute probabilities
        probs = F.softmax(action_logits, dim=-1)  # [B, H*D, num_bins]
        
        # Weighted sum over bin centers
        actions_flat = torch.sum(probs * self.bin_centers.unsqueeze(0).unsqueeze(0), dim=-1)  # [B, H*D]
        
        # Reshape to [B, H, D]
        B = actions_flat.shape[0]
        actions = actions_flat.reshape(B, self.action_horizon, self.action_dim)
        
        return actions


class PI0RLWrapper(nn.Module):
    """
    RL-compatible wrapper for official PI0Pytorch model
    
    This wrapper bridges OpenPI's flow matching training with SimpleVLA's
    token-based RL framework by:
    1. Using official PI0Pytorch for diffusion-based action generation
    2. Adding learnable action tokenization for PPO training
    3. Providing HuggingFace-compatible interfaces
    """
    
    def __init__(self, config: Pi0Config, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # Create official PI0Pytorch model
        self.pi0_model = PI0Pytorch(config)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Action tokenizer for RL training
        self.action_tokenizer = ActionTokenizer(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            num_bins=256,
            temperature=0.1
        )
        
        # Store num diffusion steps for inference
        self.num_inference_steps = 10
        
        # Normalization stats (loaded externally)
        self.norm_stats = None
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        import safetensors.torch
        
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(self.pi0_model, model_path)
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: No checkpoint found at {model_path}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.pi0_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.pi0_model.gradient_checkpointing_disable()
    
    def prepare_observation(self, pixel_values, input_ids, attention_mask, state=None, **kwargs):
        """
        Convert SimpleVLA format to OpenPI Observation format
        
        Args:
            pixel_values: [B, C, H, W] or [B, N_imgs, C, H, W] images
            input_ids: [B, L] tokenized text (from PaliGemma tokenizer)
            attention_mask: [B, L] attention mask
            state: [B, D] optional proprioceptive state
        
        Returns:
            observation: OpenPI Observation object
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Handle multiple images
        if pixel_values.ndim == 5:
            # [B, N_imgs, C, H, W] -> dict of images
            images = {
                f'image_{i}': pixel_values[:, i]
                for i in range(pixel_values.shape[1])
            }
            image_masks = {
                f'image_{i}': torch.ones(batch_size, dtype=torch.bool, device=device)
                for i in range(pixel_values.shape[1])
            }
        else:
            # Single image
            images = {'base_0_rgb': pixel_values}
            image_masks = {'base_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device)}
        
        # Default state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.config.action_dim, device=device)
        
        # Use provided tokenized input
        # NOTE: Tokenization is now done externally by PaliGemma tokenizer in verl/utils/tokenizer.py
        tokenized_prompts = input_ids
        prompt_masks = attention_mask.bool()
        
        # Create OpenPI Observation object
        observation = openpi_model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompts,
            tokenized_prompt_mask=prompt_masks
        )
        
        return observation
    
    def forward(self, pixel_values, input_ids, attention_mask=None, state=None, 
                actions=None, return_loss=False, return_continuous=False, **kwargs):
        """
        Forward pass compatible with SimpleVLA
        
        Args:
            pixel_values: [B, C, H, W] images
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask
            state: [B, D] optional proprioceptive state
            actions: [B, H, D] ground truth actions (for training)
            return_loss: whether to return flow matching loss
            return_continuous: whether to return continuous actions
        
        Returns:
            dict with logits and optionally loss/continuous_actions
        """
        # Prepare observation
        observation = self.prepare_observation(
            pixel_values, input_ids, attention_mask, state, **kwargs
        )
        
        if return_loss and actions is not None:
            # Training mode: compute flow matching loss
            losses = self.pi0_model(observation, actions)
            loss = losses.mean()
            
            # Also generate actions for tokenization (detached)
            with torch.no_grad():
                continuous_actions = self.pi0_model.sample_actions(
                    device=pixel_values.device,
                    observation=observation,
                    num_steps=self.num_inference_steps
                )
            
            # Tokenize for RL training
            action_logits = self.action_tokenizer.tokenize(continuous_actions)
            
            result = {
                'logits': action_logits,
                'loss': loss
            }
            
            if return_continuous:
                result['continuous_actions'] = continuous_actions
            
            return result
        
        else:
            # Inference mode: sample actions using diffusion
            with torch.no_grad():
                continuous_actions = self.pi0_model.sample_actions(
                    device=pixel_values.device,
                    observation=observation,
                    num_steps=self.num_inference_steps
                )
            
            # Tokenize actions (this part has gradients for the tokenizer)
            action_logits = self.action_tokenizer.tokenize(continuous_actions.detach())
            
            result = {'logits': action_logits}
            
            if return_continuous:
                result['continuous_actions'] = continuous_actions
            
            return result
    
    def compute_log_probs(self, pixel_values, input_ids, attention_mask, 
                          action_tokens, state=None, **kwargs):
        """
        Compute log probabilities for given action tokens (needed for PPO)
        
        Args:
            pixel_values: [B, C, H, W] images
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask
            action_tokens: [B, H*D] action token indices
            state: [B, D] optional proprioceptive state
        
        Returns:
            log_probs: [B, H*D] log probabilities
        """
        # Forward pass to get logits
        outputs = self.forward(pixel_values, input_ids, attention_mask, state, **kwargs)
        logits = outputs['logits']  # [B, H*D, num_bins]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, H*D, num_bins]
        
        # Gather log probs for selected actions
        selected_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=action_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [B, H*D]
        
        return selected_log_probs
    
    def generate(self, pixel_values, input_ids, attention_mask=None, state=None,
                 do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=None, **kwargs):
        """
        Generate action tokens (compatible with HuggingFace interface)
        
        Args:
            pixel_values: [B, C, H, W] images
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask
            state: [B, D] optional proprioceptive state
            do_sample: whether to sample from distribution
            temperature: sampling temperature
            top_p: nucleus sampling parameter
        
        Returns:
            action_tokens: [B, H*D] sampled action tokens
        """
        # Get action logits
        outputs = self.forward(pixel_values, input_ids, attention_mask, state, **kwargs)
        logits = outputs['logits']  # [B, H*D, num_bins]
        
        if do_sample:
            # Apply temperature
            logits = logits / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            
            # Optional: nucleus sampling (top-p)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Set filtered logits to -inf
                logits_filtered = logits.clone()
                logits_filtered[sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)] = float('-inf')
                probs = F.softmax(logits_filtered, dim=-1)
            
            # Sample
            action_tokens = torch.multinomial(
                probs.view(-1, logits.shape[-1]), 
                num_samples=1
            ).view(logits.shape[0], -1)
        else:
            # Greedy decoding
            action_tokens = logits.argmax(dim=-1)  # [B, H*D]
        
        return action_tokens
    
    def predict_action(self, pixel_values, input_ids, attention_mask=None, 
                      state=None, unnorm_key=None, **kwargs):
        """
        Predict continuous actions (for evaluation/deployment)
        
        Args:
            pixel_values: [B, C, H, W] images
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask
            state: [B, D] optional proprioceptive state
            unnorm_key: key for unnormalization stats
        
        Returns:
            actions: [B, H, D] unnormalized continuous actions
        """
        # Prepare observation
        observation = self.prepare_observation(
            pixel_values, input_ids, attention_mask, state, **kwargs
        )
        
        # Sample actions
        with torch.no_grad():
            actions = self.pi0_model.sample_actions(
                device=pixel_values.device,
                observation=observation,
                num_steps=self.num_inference_steps
            )
        
        # Unnormalize if stats available
        if self.norm_stats is not None and unnorm_key is not None:
            actions = self._unnormalize_actions(actions, unnorm_key)
        
        return actions
    
    def _unnormalize_actions(self, actions, unnorm_key):
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


def create_pi0_rl_model(model_path: str, config_name: str, action_dim: int = 7, 
                        action_horizon: int = 8, device: str = "cuda"):
    """
    Factory function to create a PI0 model for RL training
    
    Args:
        model_path: path to checkpoint directory
        config_name: config name (e.g., "pi0_libero", "pi05_libero")
        action_dim: action dimension
        action_horizon: action horizon
        device: device to load model on
    
    Returns:
        PI0RLWrapper instance
    """
    # Determine if pi05 based on config name
    pi05 = "pi05" in config_name or "pi_05" in config_name
    
    # Create config
    config = Pi0Config(
        pi05=pi05,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=200 if pi05 else 48,
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m"
    )
    
    # Create model
    model = PI0RLWrapper(config, checkpoint_path=model_path)
    model = model.to(device)
    
    return model


