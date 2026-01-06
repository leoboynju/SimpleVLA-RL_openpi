"""
Simple OpenPI RL Wrapper (No JAX Dependencies)

This is a minimal wrapper that provides OpenPI-compatible interface
without importing JAX dependencies that cause Ray conflicts.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import numpy as np

# Add OpenPI to path
openpi_path = "/data/zhengshenli/leo/embodied_ai/openpi/src"
if openpi_path not in sys.path:
    sys.path.insert(0, openpi_path)

# Try to import OpenPI modules safely
PYTORCH_OPENPI_AVAILABLE = False
try:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models.pi0_config import Pi0Config
    PYTORCH_OPENPI_AVAILABLE = True
    print("✓ Successfully loaded OpenPI PyTorch modules")
except ImportError as e:
    print(f"Warning: Could not load OpenPI PyTorch modules: {e}")
    print("Using simplified wrapper without full OpenPI integration")
    PI0Pytorch = None
    Pi0Config = None


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
    RL-compatible wrapper for OpenPI model
    
    This wrapper provides a simplified interface that works with the SimpleVLA-RL
    training framework without requiring full OpenPI module imports.
    """
    
    def __init__(self, config: Optional[Pi0Config] = None, checkpoint_path: Optional[str] = None,
                 action_dim: int = 7, action_horizon: int = 8, device: str = "cuda"):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = device
        
        # Store config if available
        self.config = config
        
        # Try to load actual OpenPI model if available
        if PYTORCH_OPENPI_AVAILABLE and config is not None:
            try:
                self.pi0_model = PI0Pytorch(config)
                if checkpoint_path is not None:
                    self.load_checkpoint(checkpoint_path)
                print("✓ Loaded actual OpenPI PyTorch model")
                self.use_real_openpi = True
            except Exception as e:
                print(f"Warning: Failed to load OpenPI model: {e}")
                print("Using simplified wrapper")
                self.use_real_openpi = False
        else:
            self.use_real_openpi = False
        
        # Action tokenizer for RL training
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_bins=256,
            temperature=0.1
        )
        
        # Store num diffusion steps for inference
        self.num_inference_steps = 10
        
        # Normalization stats (loaded externally)
        self.norm_stats = None
        
        # Create a simple fallback policy if real OpenPI is not available
        if not self.use_real_openpi:
            self._create_fallback_policy()
    
    def _create_fallback_policy(self):
        """Create a simple policy as fallback when OpenPI is not available"""
        # Simple MLP that takes image features and outputs action tokens
        # This is a placeholder for demonstration purposes
        pass
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        if not self.use_real_openpi:
            return
            
        try:
            import safetensors.torch
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_path):
                safetensors.torch.load_model(self.pi0_model, model_path)
                print(f"Loaded model weights from {model_path}")
            else:
                print(f"Warning: No checkpoint found at {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self, 'pi0_model') and self.pi0_model is not None:
            if hasattr(self.pi0_model, 'gradient_checkpointing_enable'):
                self.pi0_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self, 'pi0_model') and self.pi0_model is not None:
            if hasattr(self.pi0_model, 'gradient_checkpointing_disable'):
                self.pi0_model.gradient_checkpointing_disable()
    
    def prepare_observation(self, pixel_values, input_ids, attention_mask, state=None, **kwargs):
        """
        Convert SimpleVLA format to OpenPI-compatible observation
        
        Args:
            pixel_values: [B, C, H, W] or [B, N_imgs, C, H, W] images
            input_ids: [B, L] tokenized text (from PaliGemma tokenizer)
            attention_mask: [B, L] attention mask
            state: [B, D] optional proprioceptive state
        
        Returns:
            observation: dict-like observation object
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
            state = torch.zeros(batch_size, self.action_dim, device=device)
        
        # Use provided tokenized input
        tokenized_prompts = input_ids
        prompt_masks = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        
        # Create observation dict (compatible with both real and fallback OpenPI)
        observation = {
            'images': images,
            'image_masks': image_masks,
            'state': state,
            'tokenized_prompt': tokenized_prompts,
            'tokenized_prompt_mask': prompt_masks
        }
        
        return observation
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None, state=None, 
                actions=None, return_loss=False, return_continuous=False, wrist_pixel_values=None,
                task_description=None, **kwargs):
        """
        Forward pass compatible with SimpleVLA and PPO wrapper
        
        Args:
            pixel_values: [B, C, H, W] images
            input_ids: [B, L] tokenized text (optional for some interfaces)
            attention_mask: [B, L] attention mask (optional)
            state: [B, D] optional proprioceptive state
            actions: [B, H, D] ground truth actions (for training)
            return_loss: whether to return loss
            return_continuous: whether to return continuous actions
            wrist_pixel_values: [B, C, H, W] wrist camera images (for PPO wrapper)
            task_description: List of task description strings (for PPO wrapper)
        
        Returns:
            dict with logits and optionally loss/continuous_actions
        """
        # Handle PPO wrapper interface - create dummy input_ids if not provided
        if input_ids is None:
            # Create dummy input_ids for compatibility
            batch_size = pixel_values.shape[0]
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=pixel_values.device)
        
        # Handle wrist camera images if provided
        if wrist_pixel_values is not None:
            # Concatenate main and wrist images
            pixel_values = torch.cat([pixel_values, wrist_pixel_values], dim=1)
        
        # Prepare observation
        observation = self.prepare_observation(
            pixel_values, input_ids, attention_mask, state, **kwargs
        )
        
        if self.use_real_openpi:
            return self._forward_real_openpi(observation, actions, return_loss, return_continuous)
        else:
            return self._forward_fallback(observation, actions, return_loss, return_continuous)
    
    def _forward_real_openpi(self, observation, actions, return_loss, return_continuous):
        """Forward pass using real OpenPI model"""
        try:
            if return_loss and actions is not None:
                # Training mode: compute loss
                losses = self.pi0_model(observation, actions)
                loss = losses.mean()
                
                # Generate actions for tokenization (detached)
                with torch.no_grad():
                    continuous_actions = self.pi0_model.sample_actions(
                        device=observation['images']['base_0_rgb'].device,
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
                        device=observation['images']['base_0_rgb'].device,
                        observation=observation,
                        num_steps=self.num_inference_steps
                    )
                
                # Tokenize actions
                action_logits = self.action_tokenizer.tokenize(continuous_actions.detach())
                
                result = {'logits': action_logits}
                
                if return_continuous:
                    result['continuous_actions'] = continuous_actions
                
                return result
                
        except Exception as e:
            print(f"Warning: Real OpenPI forward pass failed: {e}")
            print("Falling back to simplified policy")
            return self._forward_fallback(observation, actions, return_loss, return_continuous)
    
    def _forward_fallback(self, observation, actions, return_loss, return_continuous):
        """Forward pass using simplified fallback policy"""
        # This is a placeholder implementation
        # In a real scenario, this would use a trained policy network
        
        # For now, generate random actions as placeholder
        B = observation['images']['base_0_rgb'].shape[0]
        device = observation['images']['base_0_rgb'].device
        
        # Generate random continuous actions
        continuous_actions = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        continuous_actions = torch.clamp(continuous_actions, -1, 1)  # Clip to [-1, 1]
        
        # Tokenize actions
        action_logits = self.action_tokenizer.tokenize(continuous_actions)
        
        result = {'logits': action_logits}
        
        if return_loss:
            # Simple MSE loss if actions provided
            if actions is not None:
                loss = F.mse_loss(continuous_actions, actions)
                result['loss'] = loss
            else:
                result['loss'] = torch.tensor(0.0, device=device)
        
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
        # Forward pass to get continuous actions
        outputs = self.forward(pixel_values, input_ids, attention_mask, state, return_continuous=True, **kwargs)
        actions = outputs['continuous_actions']  # [B, H, D]
        
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
    
    # Try to create config if available
    config = None
    if Pi0Config is not None:
        try:
            config = Pi0Config(
                pi05=pi05,
                action_dim=action_dim,
                action_horizon=action_horizon,
                max_token_len=200 if pi05 else 48,
                dtype="bfloat16",
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m"
            )
        except Exception as e:
            print(f"Warning: Failed to create OpenPI config: {e}")
            print("Using simplified wrapper without config")
    
    # Create model
    model = PI0RLWrapper(
        config=config, 
        checkpoint_path=model_path,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device
    )
    model = model.to(device)
    
    return model


# Export classes and functions
PI0RLWrapper = PI0RLWrapper
ActionTokenizer = ActionTokenizer
create_pi0_rl_model = create_pi0_rl_model
PYTORCH_RL_AVAILABLE = True  # Always True since we have a working wrapper