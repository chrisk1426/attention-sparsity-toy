"""
Phase 3: Spectral Computation

Performs eigen-decomposition and spectral analysis of circuit matrices.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class SpectralAnalyzer:
    """
    Performs spectral analysis on circuit matrices.
    Focuses on eigenvalue distributions, especially near zero.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def compute_eigen_decomposition(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalue decomposition of a matrix.
        Uses GPU-accelerated operations when available.
        
        Args:
            matrix: Input matrix [vocab_size, vocab_size]
            k: Number of top eigenvalues to compute (None = all)
            
        Returns:
            eigenvalues: Eigenvalues sorted in descending order
            eigenvectors: Corresponding eigenvectors [vocab_size, k or vocab_size]
        """
        # Ensure matrix is a torch tensor on the correct device
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        else:
            matrix = matrix.to(self.device)
        
        # Check if matrix is symmetric (within numerical tolerance)
        is_symmetric = torch.allclose(matrix, matrix.T, atol=1e-6)
        
        if is_symmetric:
            # Use eigendecomposition for symmetric matrices (faster and more accurate)
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            # Sort in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            # Use SVD for non-symmetric matrices
            # SVD: matrix = U @ diag(s) @ V^T
            U, s, Vt = torch.linalg.svd(matrix, full_matrices=False)
            eigenvalues = s
            eigenvectors = U
        
        # Select top k if specified
        if k is not None and k < len(eigenvalues):
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
        
        return eigenvalues, eigenvectors
    
    def analyze_eigenvalue_distribution(
        self,
        eigenvalues: torch.Tensor,
        bins: int = 100,
        focus_zero: bool = True,
        zero_threshold: float = 1e-6
    ) -> Dict:
        """
        Analyze the distribution of eigenvalues, focusing on tail near zero.
        
        Args:
            eigenvalues: Array of eigenvalues
            bins: Number of bins for histogram
            focus_zero: Whether to focus analysis on values near zero
            zero_threshold: Threshold for considering eigenvalues "near zero"
            
        Returns:
            Dictionary with statistics and distribution info
        """
        eigenvals_np = eigenvalues.detach().cpu().numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues
        
        stats = {
            'total': len(eigenvals_np),
            'mean': np.mean(eigenvals_np),
            'std': np.std(eigenvals_np),
            'min': np.min(eigenvals_np),
            'max': np.max(eigenvals_np),
            'median': np.median(eigenvals_np),
        }
        
        if focus_zero:
            near_zero = np.abs(eigenvals_np) < zero_threshold
            stats['near_zero_count'] = np.sum(near_zero)
            stats['near_zero_ratio'] = np.sum(near_zero) / len(eigenvals_np)
            stats['zero_threshold'] = zero_threshold
        
        # Compute density around zero
        if focus_zero:
            zero_region = eigenvals_np[np.abs(eigenvals_np) < 10 * zero_threshold]
            stats['zero_region_density'] = len(zero_region) / len(eigenvals_np)
        
        return stats
    
    def plot_eigenvalue_distribution(
        self,
        eigenvalues: torch.Tensor,
        title: str = "Eigenvalue Distribution",
        save_path: Optional[str] = None,
        focus_zero: bool = True,
        log_scale: bool = True
    ):
        """
        Plot the distribution of eigenvalues.
        
        Args:
            eigenvalues: Array of eigenvalues
            title: Plot title
            save_path: Path to save figure (optional)
            focus_zero: Whether to create zoomed plot near zero
            log_scale: Whether to use log scale for y-axis
        """
        eigenvals_np = eigenvalues.detach().cpu().numpy() if isinstance(eigenvalues, torch.Tensor) else eigenvalues
        
        fig, axes = plt.subplots(1, 2 if focus_zero else 1, figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Full distribution
        ax = axes[0]
        ax.hist(eigenvals_np, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Eigenvalue')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} - Full Distribution')
        ax.axvline(x=0, color='r', linestyle='--', label='Zero')
        ax.legend()
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Focus on zero region
        if focus_zero and len(axes) > 1:
            ax = axes[1]
            zero_region = eigenvals_np[np.abs(eigenvals_np) < np.percentile(np.abs(eigenvals_np), 10)]
            ax.hist(zero_region, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.set_xlabel('Eigenvalue')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} - Near Zero (Tail Distribution)')
            ax.axvline(x=0, color='r', linestyle='--', label='Zero')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_matrix_rank(
        self,
        matrix: torch.Tensor,
        threshold: float = 1e-6
    ) -> int:
        """
        Compute the effective rank of a matrix.
        Uses GPU-accelerated SVD when available.
        
        Args:
            matrix: Input matrix
            threshold: Threshold for considering singular values as zero
            
        Returns:
            Effective rank (number of singular values above threshold)
        """
        # Ensure matrix is on the correct device
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        else:
            matrix = matrix.to(self.device)
        
        # Compute singular values using GPU-accelerated SVD
        singular_values = torch.linalg.svdvals(matrix)
        rank = int(torch.sum(singular_values > threshold).item())
        
        return rank
