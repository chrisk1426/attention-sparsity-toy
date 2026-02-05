"""
Phase 1: Theoretical Framework & Definitions

Implements the QK and OV circuits as defined in the Mathematical Framework for Transformers.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


class CircuitAnalyzer:
    """
    Analyzes transformer attention circuits:
    - QK Circuit: The "Where" - determines attention patterns
    - OV Circuit: The "What" - determines what information is moved
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def compute_qk_circuit(
        self,
        W_E: torch.Tensor,
        W_Q: torch.Tensor,
        W_K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the full QK circuit matrix.
        
        Formula: W_QK_Full = W_E_transpose * W_QK_head * W_E
        where W_QK_head = W_Q * W_K_transpose
        
        Args:
            W_E: Embedding matrix [vocab_size, d_model]
            W_Q: Query weight matrix [d_model, d_head] or [d_model, d_model] for full
            W_K: Key weight matrix [d_model, d_head] or [d_model, d_model] for full
            
        Returns:
            W_QK_Full: Full QK circuit [vocab_size, vocab_size]
        """
        # Compute W_QK_head = W_Q * W_K^T
        if len(W_Q.shape) == 2 and W_Q.shape[1] < W_Q.shape[0]:
            # Multi-head case: [d_model, d_head]
            W_QK_head = torch.matmul(W_Q, W_K.transpose(-2, -1))
        else:
            # Single head or full: [d_model, d_model]
            W_QK_head = torch.matmul(W_Q, W_K.transpose(-2, -1))
        
        # Compute full circuit: W_E * W_QK_head * W_E^T
        # This gives [vocab, d_model] @ [d_model, d_model] @ [d_model, vocab] = [vocab, vocab]
        W_QK_Full = torch.matmul(
            torch.matmul(W_E, W_QK_head),
            W_E.transpose(-2, -1)
        )
        
        return W_QK_Full
    
    def compute_ov_circuit(
        self,
        W_E: torch.Tensor,
        W_O: torch.Tensor,
        W_V: torch.Tensor,
        W_U: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the full OV circuit matrix.
        
        Formula: W_OV_Full = W_U * W_OV_head * W_E
        where W_OV_head = W_O * W_V
        
        Args:
            W_E: Embedding matrix [vocab_size, d_model]
            W_O: Output weight matrix [d_model, d_head] or [d_model, d_model]
            W_V: Value weight matrix [d_model, d_head] or [d_model, d_model]
            W_U: Unembedding matrix [d_model, vocab_size]
            
        Returns:
            W_OV_Full: Full OV circuit [vocab_size, vocab_size]
        """
        # Compute W_OV_head = W_V * W_O^T (or W_O @ W_V^T depending on shape)
        # W_V: [d_model, d_head], W_O: [d_model, d_head] -> W_OV_head: [d_model, d_model]
        if len(W_O.shape) == 2 and W_O.shape[1] < W_O.shape[0]:
            # Multi-head case: W_V @ W_O^T gives [d_model, d_model]
            W_OV_head = torch.matmul(W_V, W_O.transpose(-2, -1))
        else:
            # Full case: already [d_model, d_model]
            W_OV_head = torch.matmul(W_V, W_O.transpose(-2, -1))

        # Compute full circuit: W_U * W_OV_head * W_E
        # W_U: [vocab, d_model], W_OV_head: [d_model, d_model], W_E: [vocab, d_model]
        # Result: [vocab, d_model] @ [d_model, d_model] @ [d_model, vocab] = [vocab, vocab]
        W_OV_Full = torch.matmul(
            torch.matmul(W_U, W_OV_head),
            W_E.transpose(-2, -1)
        )
        
        return W_OV_Full
    
    def compute_circuits_for_head(
        self,
        W_E: torch.Tensor,
        W_U: torch.Tensor,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        W_O: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both QK and OV circuits for a single attention head.
        
        Returns:
            W_QK_Full, W_OV_Full: Both circuit matrices [vocab_size, vocab_size]
        """
        W_QK_Full = self.compute_qk_circuit(W_E, W_Q, W_K)
        W_OV_Full = self.compute_ov_circuit(W_E, W_O, W_V, W_U)
        
        return W_QK_Full, W_OV_Full
