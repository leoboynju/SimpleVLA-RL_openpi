"""
OpenPI model integration for SimpleVLA-RL

This package provides wrappers to integrate OpenPI models (π₀ and π₀.₅)
with the SimpleVLA-RL training framework.

Location: verl/utils/vla_utils/openpi/

IMPORTANT: This package avoids importing from openpi.shared due to orbax/tensorstore
conflicts that cause SIGABRT when used with Ray. Use PI0RLWrapper for PPO training.
"""

# PaliGemma tokenizer (copied from OpenPI) - imported first to avoid JAX deps
try:
    from .paligemma_tokenizer import PaligemmaTokenizer
    PALIGEMMA_TOKENIZER_AVAILABLE = True
except ImportError as e:
    PALIGEMMA_TOKENIZER_AVAILABLE = False
    print(f"Warning: PaligemmaTokenizer not available: {e}")
    PaligemmaTokenizer = None

# Legacy JAX-based wrapper - DEPRECATED, disabled to avoid orbax/tensorstore crashes
# The JAX-based OpenPI (openpi.shared.download) causes SIGABRT when imported
# Use PI0RLWrapper (pi0_rl_wrapper.py) instead for PPO training
LEGACY_JAX_AVAILABLE = False
OpenPIModel = None
create_openpi_model = None
print("Note: Legacy JAX-based OpenPI is disabled (requires orbax which conflicts with Ray). "
      "Use PI0RLWrapper from pi0_rl_wrapper.py for PPO training.")

# Official PyTorch implementation wrapper (RL-compatible)
# This is the RECOMMENDED wrapper for SimpleVLA-RL training
PYTORCH_RL_AVAILABLE = False
try:
    # Try to use the simplified wrapper first (no JAX dependencies)
    from .pi0_rl_wrapper_simple import PI0RLWrapper, ActionTokenizer, create_pi0_rl_model
    PYTORCH_RL_AVAILABLE = True
    print("✓ Using simplified OpenPI RL wrapper (no JAX dependencies)")
except ImportError as e:
    PYTORCH_RL_AVAILABLE = False
    print(f"Warning: OpenPI RL wrapper not available: {e}")
    print("  - This is required for PPO training with OpenPI models")
    PI0RLWrapper = None
    ActionTokenizer = None
    create_pi0_rl_model = None

# PPO wrapper for continuous action space training
# Provides Gaussian policy wrapper for PPO training
try:
    from .ppo_wrapper import OpenPIPPOWrapper, create_openpi_ppo_wrapper
    PPO_WRAPPER_AVAILABLE = True
except ImportError as e:
    PPO_WRAPPER_AVAILABLE = False
    print(f"Warning: OpenPI PPO wrapper not available: {e}")
    OpenPIPPOWrapper = None
    create_openpi_ppo_wrapper = None

# Universal wrapper for OpenPI models (auto-detects model type)
# Provides compatibility with both direct and wrapper-based models
try:
    from .compatible_wrapper import CompatibleOpenPIWrapper, create_compatible_wrapper
    COMPATIBLE_WRAPPER_AVAILABLE = True
    print("✓ Compatible OpenPI wrapper available (supports both direct and wrapper models)")
except ImportError as e:
    COMPATIBLE_WRAPPER_AVAILABLE = False
    print(f"Warning: Compatible OpenPI wrapper not available: {e}")
    CompatibleOpenPIWrapper = None
    create_compatible_wrapper = None

__all__ = [
    'PaligemmaTokenizer',
    'PALIGEMMA_TOKENIZER_AVAILABLE',
    'OpenPIModel',
    'create_openpi_model',
    'LEGACY_JAX_AVAILABLE',
    'PI0RLWrapper',
    'ActionTokenizer',
    'create_pi0_rl_model',
    'PYTORCH_RL_AVAILABLE',
    'OpenPIPPOWrapper',
    'create_openpi_ppo_wrapper',
    'PPO_WRAPPER_AVAILABLE',
    'CompatibleOpenPIWrapper',
    'create_compatible_wrapper',
    'COMPATIBLE_WRAPPER_AVAILABLE',
]
