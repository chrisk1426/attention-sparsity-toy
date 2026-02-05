"""
Phase 4: Comparative Analysis

Compares spectral signatures between reasoning and non-reasoning models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from spectral_computation import SpectralAnalyzer


class ComparativeAnalyzer:
    """
    Compares spectral properties between different models.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.spectral_analyzer = SpectralAnalyzer(device)
    
    def compare_spectral_densities(
        self,
        eigenvalues_reasoning: torch.Tensor,
        eigenvalues_control: torch.Tensor,
        labels: Tuple[str, str] = ("Reasoning Model", "Control Model")
    ) -> Dict:
        """
        Compare spectral density distributions between two models.
        
        Args:
            eigenvalues_reasoning: Eigenvalues from reasoning model
            eigenvalues_control: Eigenvalues from control model
            labels: Labels for the two models
            
        Returns:
            Dictionary with comparison statistics
        """
        eigen_reasoning = eigenvalues_reasoning.detach().cpu().numpy() if isinstance(eigenvalues_reasoning, torch.Tensor) else eigenvalues_reasoning
        eigen_control = eigenvalues_control.detach().cpu().numpy() if isinstance(eigenvalues_control, torch.Tensor) else eigenvalues_control
        
        comparison = {
            'reasoning_stats': self.spectral_analyzer.analyze_eigenvalue_distribution(eigenvalues_reasoning),
            'control_stats': self.spectral_analyzer.analyze_eigenvalue_distribution(eigenvalues_control),
        }
        
        # Compute differences
        comparison['mean_difference'] = comparison['reasoning_stats']['mean'] - comparison['control_stats']['mean']
        comparison['std_difference'] = comparison['reasoning_stats']['std'] - comparison['control_stats']['std']
        comparison['near_zero_ratio_diff'] = (
            comparison['reasoning_stats'].get('near_zero_ratio', 0) - 
            comparison['control_stats'].get('near_zero_ratio', 0)
        )
        
        return comparison
    
    def identify_thinking_heads(
        self,
        head_eigenvalues: Dict[Tuple[int, int], torch.Tensor],
        threshold_percentile: float = 90
    ) -> List[Tuple[int, int]]:
        """
        Identify heads with the most unique spectral signatures.
        
        Args:
            head_eigenvalues: Dict mapping (layer, head) -> eigenvalues
            threshold_percentile: Percentile threshold for uniqueness
            
        Returns:
            List of (layer, head) tuples for "thinking" heads
        """
        # Compute uniqueness metric for each head
        uniqueness_scores = {}
        
        for (layer, head), eigenvals in head_eigenvalues.items():
            eigenvals_np = eigenvals.detach().cpu().numpy() if isinstance(eigenvals, torch.Tensor) else eigenvals
            
            # Uniqueness: variance in tail distribution (near zero)
            near_zero = eigenvals_np[np.abs(eigenvals_np) < np.percentile(np.abs(eigenvals_np), 10)]
            if len(near_zero) > 0:
                uniqueness = np.std(near_zero)
            else:
                uniqueness = 0
            
            uniqueness_scores[(layer, head)] = uniqueness
        
        # Find heads above threshold
        threshold = np.percentile(list(uniqueness_scores.values()), threshold_percentile)
        thinking_heads = [
            (layer, head) for (layer, head), score in uniqueness_scores.items()
            if score >= threshold
        ]
        
        return thinking_heads
    
    def interpret_eigenvalues(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        tokenizer,
        top_k: int = 10
    ) -> Dict:
        """
        Interpret eigenvalues by examining corresponding eigenvectors.
        
        Args:
            eigenvalues: Eigenvalues
            eigenvectors: Eigenvectors [vocab_size, num_eigenvectors]
            tokenizer: Tokenizer for converting indices to tokens
            top_k: Number of top tokens to extract per eigenvector
            
        Returns:
            Dictionary mapping eigenvalue -> top tokens
        """
        eigenvals_np = eigenvalues.detach().cpu().numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues
        eigenvecs_np = eigenvectors.detach().cpu().numpy() if isinstance(eigenvectors, torch.Tensor) else eigenvectors
        
        interpretations = {}
        
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvals_np, eigenvecs_np.T)):
            # Get top and bottom tokens
            top_indices = np.argsort(eigenvec)[-top_k:][::-1]
            bottom_indices = np.argsort(eigenvec)[:top_k]
            
            top_tokens = [tokenizer.decode([idx]) if hasattr(tokenizer, 'decode') else str(idx) for idx in top_indices]
            bottom_tokens = [tokenizer.decode([idx]) if hasattr(tokenizer, 'decode') else str(idx) for idx in bottom_indices]
            
            interpretations[eigenval] = {
                'top_tokens': top_tokens,
                'bottom_tokens': bottom_tokens,
                'magnitude': abs(eigenval)
            }
        
        return interpretations
    
    def plot_comparison(
        self,
        eigenvalues_reasoning: torch.Tensor,
        eigenvalues_control: torch.Tensor,
        labels: Tuple[str, str] = ("Reasoning Model", "Control Model"),
        save_path: Optional[str] = None
    ):
        """Plot side-by-side comparison of eigenvalue distributions."""
        eigen_reasoning = eigenvalues_reasoning.detach().cpu().numpy() if isinstance(eigenvalues_reasoning, torch.Tensor) else eigenvalues_reasoning
        eigen_control = eigenvalues_control.detach().cpu().numpy() if isinstance(eigenvalues_control, torch.Tensor) else eigenvalues_control
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(eigen_reasoning, bins=100, alpha=0.7, label=labels[0], color='blue')
        axes[0].set_xlabel('Eigenvalue')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{labels[0]} - Eigenvalue Distribution')
        axes[0].axvline(x=0, color='r', linestyle='--')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(eigen_control, bins=100, alpha=0.7, label=labels[1], color='orange')
        axes[1].set_xlabel('Eigenvalue')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{labels[1]} - Eigenvalue Distribution')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_rank_hypothesis(
        self,
        matrices: List[torch.Tensor],
        threshold: float = 1e-6
    ) -> Dict:
        """
        Test the null hypothesis that matrices are high-rank.
        
        Args:
            matrices: List of matrices to test
            threshold: Threshold for rank computation
            
        Returns:
            Dictionary with rank statistics
        """
        ranks = []
        full_ranks = []
        
        for matrix in matrices:
            rank = self.spectral_analyzer.compute_matrix_rank(matrix, threshold)
            full_rank = min(matrix.shape)
            ranks.append(rank)
            full_ranks.append(full_rank)
        
        return {
            'ranks': ranks,
            'full_ranks': full_ranks,
            'rank_ratios': [r / fr for r, fr in zip(ranks, full_ranks)],
            'mean_rank_ratio': np.mean([r / fr for r, fr in zip(ranks, full_ranks)]),
            'is_high_rank': np.mean([r / fr for r, fr in zip(ranks, full_ranks)]) > 0.8
        }
