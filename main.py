"""
Main execution script for Spectral Analysis of Toy Models

Implements all 5 phases of the project outline.
"""

import torch
import numpy as np
from circuit_analysis import CircuitAnalyzer
from model_loader import ModelLoader
from spectral_computation import SpectralAnalyzer
from comparative_analysis import ComparativeAnalyzer
from bias_detection import BiasDetector
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def main():
    """Main execution function implementing all 5 phases."""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # ========================================================================
    # PHASE 1: Theoretical Framework & Definitions
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Theoretical Framework & Definitions")
    print("="*80)
    
    circuit_analyzer = CircuitAnalyzer(device=device)
    print("✓ Circuit analyzer initialized")
    print("  - QK Circuit: W_QK_Full = W_E^T * W_QK_head * W_E")
    print("  - OV Circuit: W_OV_Full = W_U * W_OV_head * W_E")
    
    # ========================================================================
    # PHASE 2: Model Selection & Data Extraction
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Model Selection & Data Extraction")
    print("="*80)
    
    # Load reasoning model
    model_loader = ModelLoader(
        model_name="roneneldan/TinyStories-33M",
        device=device
    )
    model_loader.load_model()
    weights_reasoning = model_loader.extract_weights()
    print(f"✓ Reasoning model loaded and weights extracted")
    
    # Create control model (random initialization)
    print("\nCreating control model (random initialization)...")
    weights_control = model_loader.create_random_model()
    print("✓ Control model created")
    
    # ========================================================================
    # PHASE 3: Spectral Computation
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Spectral Computation")
    print("="*80)
    
    spectral_analyzer = SpectralAnalyzer(device=device)
    
    # Compute circuits for first layer, first head (if available)
    W_E = weights_reasoning['W_E']
    W_U = weights_reasoning['W_U']
    
    # Process each layer and head
    all_qk_circuits = []
    all_ov_circuits = []
    all_eigenvalues_qk = []
    all_eigenvalues_ov = []
    all_eigenvectors_qk = []
    all_eigenvectors_ov = []
    
    if 'attention_weights' in weights_reasoning and len(weights_reasoning['attention_weights']) > 0:
        layer_idx = list(weights_reasoning['attention_weights'].keys())[0]
        head_weights = weights_reasoning['attention_weights'][layer_idx]
        
        print(f"\nComputing circuits for layer {layer_idx}...")
        
        # Compute QK circuit
        W_QK_Full = circuit_analyzer.compute_qk_circuit(
            W_E, head_weights['W_Q'], head_weights['W_K']
        )
        all_qk_circuits.append(W_QK_Full)
        print(f"✓ QK circuit computed: shape {W_QK_Full.shape}")
        
        # Compute OV circuit
        W_OV_Full = circuit_analyzer.compute_ov_circuit(
            W_E, head_weights['W_O'], head_weights['W_V'], W_U
        )
        all_ov_circuits.append(W_OV_Full)
        print(f"✓ OV circuit computed: shape {W_OV_Full.shape}")
        
        # Eigen-decomposition
        print("\nComputing eigen-decomposition...")
        eigenvals_qk, eigenvecs_qk = spectral_analyzer.compute_eigen_decomposition(W_QK_Full)
        eigenvals_ov, eigenvecs_ov = spectral_analyzer.compute_eigen_decomposition(W_OV_Full)
        
        all_eigenvalues_qk.append(eigenvals_qk)
        all_eigenvalues_ov.append(eigenvals_ov)
        all_eigenvectors_qk.append(eigenvecs_qk)
        all_eigenvectors_ov.append(eigenvecs_ov)
        
        print(f"✓ QK eigenvalues: {len(eigenvals_qk)} computed")
        print(f"✓ OV eigenvalues: {len(eigenvals_ov)} computed")
        
        # Analyze distribution
        qk_stats = spectral_analyzer.analyze_eigenvalue_distribution(eigenvals_qk)
        ov_stats = spectral_analyzer.analyze_eigenvalue_distribution(eigenvals_ov)
        
        print(f"\nQK Circuit Statistics:")
        print(f"  Mean: {qk_stats['mean']:.6f}")
        print(f"  Std: {qk_stats['std']:.6f}")
        print(f"  Near zero ratio: {qk_stats.get('near_zero_ratio', 0):.4f}")
        
        print(f"\nOV Circuit Statistics:")
        print(f"  Mean: {ov_stats['mean']:.6f}")
        print(f"  Std: {ov_stats['std']:.6f}")
        print(f"  Near zero ratio: {ov_stats.get('near_zero_ratio', 0):.4f}")
        
        # Plot distributions
        print("\nGenerating plots...")
        spectral_analyzer.plot_eigenvalue_distribution(
            eigenvals_qk,
            title="QK Circuit - Eigenvalue Distribution",
            save_path="results/qk_eigenvalue_distribution.png"
        )
        spectral_analyzer.plot_eigenvalue_distribution(
            eigenvals_ov,
            title="OV Circuit - Eigenvalue Distribution",
            save_path="results/ov_eigenvalue_distribution.png"
        )
        print("✓ Plots saved to results/")
    
    # ========================================================================
    # PHASE 4: Comparative Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Comparative Analysis")
    print("="*80)
    
    comparative_analyzer = ComparativeAnalyzer(device=device)
    
    # Compute control model circuits
    W_E_control = weights_control['W_E']
    W_U_control = weights_control['W_U']
    
    if 'attention_weights' in weights_control and len(weights_control['attention_weights']) > 0:
        layer_idx = list(weights_control['attention_weights'].keys())[0]
        head_weights_control = weights_control['attention_weights'][layer_idx]
        
        W_QK_Full_control = circuit_analyzer.compute_qk_circuit(
            W_E_control, head_weights_control['W_Q'], head_weights_control['W_K']
        )
        W_OV_Full_control = circuit_analyzer.compute_ov_circuit(
            W_E_control, head_weights_control['W_O'], head_weights_control['W_V'], W_U_control
        )
        
        eigenvals_qk_control, _ = spectral_analyzer.compute_eigen_decomposition(W_QK_Full_control)
        eigenvals_ov_control, _ = spectral_analyzer.compute_eigen_decomposition(W_OV_Full_control)
        
        # Compare
        comparison_qk = comparative_analyzer.compare_spectral_densities(
            all_eigenvalues_qk[0],
            eigenvals_qk_control,
            labels=("Reasoning Model", "Control Model")
        )
        
        print("\nQK Circuit Comparison:")
        print(f"  Mean difference: {comparison_qk['mean_difference']:.6f}")
        print(f"  Near zero ratio difference: {comparison_qk.get('near_zero_ratio_diff', 0):.4f}")
        
        # Plot comparison
        comparative_analyzer.plot_comparison(
            all_eigenvalues_qk[0],
            eigenvals_qk_control,
            labels=("Reasoning Model (QK)", "Control Model (QK)"),
            save_path="results/comparison_qk.png"
        )
        
        # Rank analysis
        rank_results = comparative_analyzer.test_rank_hypothesis(
            [W_QK_Full, W_OV_Full, W_QK_Full_control, W_OV_Full_control]
        )
        print(f"\nRank Analysis:")
        print(f"  Mean rank ratio: {rank_results['mean_rank_ratio']:.4f}")
        print(f"  Is high rank: {rank_results['is_high_rank']}")
    
    # ========================================================================
    # PHASE 5: Bias Detection & Conclusion
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: Bias Detection & Conclusion")
    print("="*80)
    
    bias_detector = BiasDetector(device=device)
    
    # Example bias terms (can be customized)
    bias_terms = {
        'gender': ['male', 'female', 'man', 'woman', 'he', 'she'],
        'career': ['doctor', 'nurse', 'engineer', 'teacher']
    }
    
    if len(all_eigenvectors_ov) > 0 and model_loader.tokenizer:
        print("\nProbing for biases...")
        bias_results = bias_detector.probe_bias(
            all_eigenvectors_ov[0],
            model_loader.tokenizer,
            bias_terms
        )
        print("✓ Bias probing completed")
        
        # Visualize
        bias_detector.visualize_bias(
            bias_results,
            save_path="results/bias_detection.png"
        )
    
    # Validate eigen-analysis
    print("\nValidating eigen-analysis...")
    validation = bias_detector.validate_eigen_analysis(
        all_qk_circuits + all_ov_circuits,
        all_eigenvalues_qk + all_eigenvalues_ov,
        all_eigenvectors_qk + all_eigenvectors_ov
    )
    print(f"✓ Validation completed")
    print(f"  Mean reconstruction error: {validation['mean_reconstruction_error']:.6e}")
    print(f"  Mean sparsity ratio: {validation['mean_sparsity']:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults saved to 'results/' directory")
    print("\nConclusion: Eigen-analysis provides insights into:")
    print("  - Where attention patterns form (QK circuit)")
    print("  - What information is moved (OV circuit)")
    print("  - Spectral signatures of reasoning vs non-reasoning models")
    print("  - Potential biases encoded in eigenvectors")


if __name__ == "__main__":
    main()
