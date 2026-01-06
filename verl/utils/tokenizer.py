# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""
import warnings

__all__ = ['hf_tokenizer']


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer.

    Args:
        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    
    model = kwargs.get("model",None)
    
    if model == "openvla-oft":   
        from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        print("*********USE VLA tokenizer*************")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer=processor.tokenizer
    elif model == "openvla":
        from verl.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        print("*********USE VLA tokenizer*************")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer=processor.tokenizer
    elif model in ["pi_0", "pi_05"]:
        # OpenPI models use PaliGemma tokenizer (copied from official implementation)
        print(f"*********USE OpenPI ({model}) PaliGemma tokenizer*************")
        
        from verl.utils.vla_utils.openpi.paligemma_tokenizer import PaligemmaTokenizer
        
        base_tokenizer = PaligemmaTokenizer()
        
        # Wrap in HF-compatible interface for SimpleVLA-RL
        class OpenPITokenizerWrapper:
            """HuggingFace-compatible wrapper for PaliGemma tokenizer"""
            def __init__(self, paligemma_tokenizer, model_type):
                self._tokenizer = paligemma_tokenizer
                self.model_type = model_type  # "pi_0" or "pi_05"
                self.vocab_size = paligemma_tokenizer._tokenizer.vocab_size()
                
                # Required attributes for HF compatibility
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.bos_token = "<s>"
                
                # Additional attributes commonly used
                self.pad_token_type_id = 0
                self.eos_token_type_id = 1
                self.bos_token_type_id = 2
            
            def __call__(self, text, state=None, return_tensors=None, **kwargs):
                """
                Tokenize text (and optionally state for π₀.₅)
                
                Args:
                    text: str or list[str]
                    state: np.ndarray or None (for π₀.₅ only)
                    return_tensors: 'pt' for torch tensors, None for numpy
                """
                import numpy as np
                
                if isinstance(text, str):
                    text = [text]
                    single_input = True
                else:
                    single_input = False
                
                all_tokens = []
                all_masks = []
                
                for i, txt in enumerate(text):
                    # Get state for this sample if provided
                    sample_state = None
                    if state is not None:
                        if self.model_type == "pi_05":
                            sample_state = state[i] if len(state.shape) > 1 else state
                    
                    tokens, mask = self._tokenizer.tokenize(txt, state=sample_state)
                    all_tokens.append(tokens)
                    all_masks.append(mask)
                
                all_tokens = np.stack(all_tokens)
                all_masks = np.stack(all_masks)
                
                if return_tensors == 'pt':
                    import torch
                    all_tokens = torch.from_numpy(all_tokens).long()
                    all_masks = torch.from_numpy(all_masks).bool()
                
                if single_input:
                    all_tokens = all_tokens[0]
                    all_masks = all_masks[0]
                
                return {
                    'input_ids': all_tokens,
                    'attention_mask': all_masks
                }
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                """Encode single text to token IDs"""
                tokens, mask = self._tokenizer.tokenize(text)
                return tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
            
            def decode(self, token_ids, **kwargs):
                """Decode token IDs back to text"""
                # Handle numpy arrays, lists, or tensors
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
                elif hasattr(token_ids, 'cpu'):  # PyTorch tensor
                    token_ids = token_ids.cpu().tolist()
                elif isinstance(token_ids, np.ndarray):
                    token_ids = token_ids.tolist()
                return self._tokenizer._tokenizer.decode(token_ids)
            
            def batch_decode(self, token_ids_list, **kwargs):
                """Decode batch of token IDs"""
                return [self.decode(ids, **kwargs) for ids in token_ids_list]
        
        tokenizer = OpenPITokenizerWrapper(base_tokenizer, model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer
