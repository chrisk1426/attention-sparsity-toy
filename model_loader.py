"""
Phase 2: Model Selection & Data Extraction

Loads transformer models and extracts weights for analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
import numpy as np


class ModelLoader:
    """
    Loads transformer models and extracts weights for circuit analysis.
    """
    
    def __init__(self, model_name: str = "roneneldan/TinyStories-33M", device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract embedding, unembedding, and attention head weights.
        
        Returns:
            Dictionary containing:
            - W_E: Embedding matrix
            - W_U: Unembedding matrix
            - attention_weights: Dict of head weights per layer
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        state_dict = self.model.state_dict()
        weights = {}
        
        # Extract embedding and unembedding
        # Common naming patterns
        if 'transformer.wte.weight' in state_dict:
            weights['W_E'] = state_dict['transformer.wte.weight'].to(self.device)
        elif 'model.embed_tokens.weight' in state_dict:
            weights['W_E'] = state_dict['model.embed_tokens.weight'].to(self.device)
        else:
            # Try to find embedding layer
            for key in state_dict.keys():
                if 'embed' in key.lower() and 'weight' in key:
                    weights['W_E'] = state_dict[key].to(self.device)
                    break
        
        # Extract unembedding (usually tied to embedding in GPT models)
        if 'lm_head.weight' in state_dict:
            weights['W_U'] = state_dict['lm_head.weight'].to(self.device)
        elif 'transformer.wte.weight' in state_dict:
            # Weight tying: unembedding = embedding transpose
            weights['W_U'] = state_dict['transformer.wte.weight'].to(self.device)
        else:
            # Try to find output layer
            for key in state_dict.keys():
                if ('lm_head' in key or 'output' in key.lower()) and 'weight' in key:
                    weights['W_U'] = state_dict[key].to(self.device)
                    break
        
        # Extract attention head weights
        weights['attention_weights'] = {}
        num_layers = 0

        for key in state_dict.keys():
            if 'attn' in key.lower():
                # Count layers - handle different naming conventions
                if 'layers.' in key:
                    layer_num = int(key.split('layers.')[1].split('.')[0])
                    num_layers = max(num_layers, layer_num + 1)
                elif 'transformer.h.' in key:
                    # GPT-Neo style: transformer.h.0.attn...
                    layer_num = int(key.split('transformer.h.')[1].split('.')[0])
                    num_layers = max(num_layers, layer_num + 1)
        
        # Extract Q, K, V, O weights for each layer and head
        for layer_idx in range(num_layers):
            layer_weights = {}
            
            # Pattern matching for different architectures
            patterns = [
                f'layers.{layer_idx}.attention',
                f'transformer.h.{layer_idx}.attn',
                f'layers.{layer_idx}.self_attn'
            ]
            
            for pattern in patterns:
                q_key = None
                k_key = None
                v_key = None
                o_key = None

                for key in state_dict.keys():
                    # Only match weight matrices, not biases
                    if pattern in key and key.endswith('.weight'):
                        if 'q_proj' in key or 'query' in key.lower():
                            q_key = key
                        elif 'k_proj' in key or 'key' in key.lower():
                            k_key = key
                        elif 'v_proj' in key or 'value' in key.lower():
                            v_key = key
                        elif 'o_proj' in key or 'out_proj' in key.lower() or 'dense' in key.lower():
                            o_key = key
                
                if q_key and k_key and v_key and o_key:
                    layer_weights['W_Q'] = state_dict[q_key].to(self.device)
                    layer_weights['W_K'] = state_dict[k_key].to(self.device)
                    layer_weights['W_V'] = state_dict[v_key].to(self.device)
                    layer_weights['W_O'] = state_dict[o_key].to(self.device)
                    break
            
            if layer_weights:
                weights['attention_weights'][layer_idx] = layer_weights
        
        print(f"Extracted weights from {num_layers} layers")
        return weights
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        return len(self.tokenizer)
    
    def create_random_model(self) -> Dict[str, torch.Tensor]:
        """
        Create a randomly initialized model with same architecture.
        Useful for control experiments.
        Returns random weights without modifying the original model.
        """
        if self.model is None:
            raise ValueError("Load model first to get architecture.")

        # Create random weights with same shapes (without modifying original model)
        random_weights = {}
        state_dict = self.model.state_dict()

        # Create random embedding
        if 'W_E' in self.extract_weights():
            original_W_E = self.extract_weights()['W_E']
            random_weights['W_E'] = torch.randn_like(original_W_E) * 0.02

        # Create random unembedding
        if 'W_U' in self.extract_weights():
            original_W_U = self.extract_weights()['W_U']
            random_weights['W_U'] = torch.randn_like(original_W_U) * 0.02

        # Create random attention weights
        original_weights = self.extract_weights()
        random_weights['attention_weights'] = {}

        if 'attention_weights' in original_weights:
            for layer_idx, layer_weights in original_weights['attention_weights'].items():
                random_weights['attention_weights'][layer_idx] = {
                    key: torch.randn_like(val) * 0.02
                    for key, val in layer_weights.items()
                }

        return random_weights
