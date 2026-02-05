"""
Phase 5: Bias Detection & Conclusion

Uses eigenvectors to detect biases and validate eigen-analysis as an interpretation tool.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from spectral_computation import SpectralAnalyzer


class BiasDetector:
    """
    Detects biases using eigenvector analysis.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.spectral_analyzer = SpectralAnalyzer(device)
    
    def probe_bias(
        self,
        eigenvectors: torch.Tensor,
        tokenizer,
        bias_terms: Dict[str, List[str]],
        top_k: int = 20
    ) -> Dict:
        """
        Probe for biases using eigenvectors.
        
        Args:
            eigenvectors: Eigenvectors [vocab_size, num_eigenvectors]
            tokenizer: Tokenizer for token conversion
            bias_terms: Dictionary mapping bias category -> list of terms
                       e.g., {'gender': ['male', 'female'], 'career': ['doctor', 'nurse']}
            top_k: Number of top tokens to examine per eigenvector
            
        Returns:
            Dictionary with bias detection results
        """
        eigenvecs_np = eigenvectors.detach().cpu().numpy() if isinstance(eigenvectors, torch.Tensor) else eigenvectors
        
        # Tokenize bias terms
        bias_token_ids = {}
        for category, terms in bias_terms.items():
            bias_token_ids[category] = []
            for term in terms:
                if hasattr(tokenizer, 'encode'):
                    token_ids = tokenizer.encode(term, add_special_tokens=False)
                    bias_token_ids[category].extend(token_ids)
                else:
                    # Fallback: try to find tokens manually
                    pass
        
        # Analyze each eigenvector
        bias_results = {}
        
        for i, eigenvec in enumerate(eigenvecs_np.T):
            # Get top and bottom tokens
            top_indices = np.argsort(eigenvec)[-top_k:][::-1]
            bottom_indices = np.argsort(eigenvec)[:top_k]
            
            # Check if bias terms appear in top/bottom
            eigenvec_bias = {}
            for category, token_ids in bias_token_ids.items():
                top_matches = sum(1 for idx in top_indices if idx in token_ids)
                bottom_matches = sum(1 for idx in bottom_indices if idx in token_ids)
                
                eigenvec_bias[category] = {
                    'top_matches': top_matches,
                    'bottom_matches': bottom_matches,
                    'total_matches': top_matches + bottom_matches,
                    'bias_score': (top_matches - bottom_matches) / len(token_ids) if len(token_ids) > 0 else 0
                }
            
            bias_results[i] = eigenvec_bias
        
        return bias_results
    
    def visualize_bias(
        self,
        bias_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize bias detection results.
        
        Args:
            bias_results: Results from probe_bias
            save_path: Path to save figure
        """
        # Extract bias scores per category
        categories = set()
        for eigenvec_results in bias_results.values():
            categories.update(eigenvec_results.keys())
        
        fig, axes = plt.subplots(len(categories), 1, figsize=(12, 4 * len(categories)))
        if len(categories) == 1:
            axes = [axes]
        
        for idx, category in enumerate(sorted(categories)):
            scores = [
                eigenvec_results[category]['bias_score']
                for eigenvec_results in bias_results.values()
                if category in eigenvec_results
            ]
            
            axes[idx].bar(range(len(scores)), scores)
            axes[idx].set_xlabel('Eigenvector Index')
            axes[idx].set_ylabel('Bias Score')
            axes[idx].set_title(f'Bias Detection: {category}')
            axes[idx].axhline(y=0, color='r', linestyle='--')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def validate_eigen_analysis(
        self,
        circuit_matrices: List[torch.Tensor],
        eigenvalues_list: List[torch.Tensor],
        eigenvectors_list: List[torch.Tensor]
    ) -> Dict:
        """
        Validate that eigen-analysis is a viable tool for interpretation.
        
        Args:
            circuit_matrices: Original circuit matrices
            eigenvalues_list: Computed eigenvalues for each matrix
            eigenvectors_list: Computed eigenvectors for each matrix
            
        Returns:
            Dictionary with validation metrics
        """
        validation = {
            'reconstruction_errors': [],
            'rank_preservation': [],
            'sparsity_ratios': []
        }
        
        for matrix, eigenvals, eigenvecs in zip(circuit_matrices, eigenvalues_list, eigenvectors_list):
            # Check reconstruction error
            matrix_np = matrix.detach().cpu().numpy() if isinstance(matrix, torch.Tensor) else matrix
            eigenvals_np = eigenvals.detach().cpu().numpy() if isinstance(eigenvals, torch.Tensor) else eigenvals
            eigenvecs_np = eigenvecs.detach().cpu().numpy() if isinstance(eigenvecs, torch.Tensor) else eigenvecs
            
            # Reconstruct matrix: M = V * diag(λ) * V^T
            reconstructed = eigenvecs_np @ np.diag(eigenvals_np) @ eigenvecs_np.T
            error = np.mean(np.abs(matrix_np - reconstructed))
            validation['reconstruction_errors'].append(error)
            
            # Check rank preservation
            original_rank = self.spectral_analyzer.compute_matrix_rank(matrix)
            eigen_rank = np.sum(np.abs(eigenvals_np) > 1e-6)
            validation['rank_preservation'].append(abs(original_rank - eigen_rank))
            
            # Check sparsity (ratio of near-zero eigenvalues)
            near_zero = np.sum(np.abs(eigenvals_np) < 1e-6) / len(eigenvals_np)
            validation['sparsity_ratios'].append(near_zero)
        
        validation['mean_reconstruction_error'] = np.mean(validation['reconstruction_errors'])
        validation['mean_rank_preservation'] = np.mean(validation['rank_preservation'])
        validation['mean_sparsity'] = np.mean(validation['sparsity_ratios'])
        
        return validation
