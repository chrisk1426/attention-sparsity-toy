"""
Main execution script for Spectral Analysis of Toy Models

Implements Phases 1-4 of the project outline.
"""

import torch
import numpy as np
from circuit_analysis import CircuitAnalyzer
from model_loader import ModelLoader
from spectral_computation import SpectralAnalyzer
from comparative_analysis import ComparativeAnalyzer
import matplotlib.pyplot as plt
from typing import Dict, List
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
INCLUDE_CONTROL_MODEL = False  # Set to True to include control model comparison


def log(msg):
    """Print with timestamp and flush immediately."""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def main():
    """Main execution function implementing Phases 1-4."""

    total_start = time.time()

    # Setup - use CPU for model/circuits, GPU only for SVD
    svd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device = 'cpu'  # Keep model and circuits on CPU
    log(f"Model device: {model_device}, SVD device: {svd_device}")

    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"GPU memory free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # ========================================================================
    # PHASE 1: Theoretical Framework & Definitions
    # ========================================================================
    log("=" * 60)
    log("PHASE 1: Theoretical Framework & Definitions")
    log("=" * 60)

    circuit_analyzer = CircuitAnalyzer(device=model_device)
    log("✓ Circuit analyzer initialized")
    log("  - QK Circuit: W_QK_Full = W_E^T * W_QK_head * W_E")
    log("  - OV Circuit: W_OV_Full = W_U * W_OV_head * W_E")

    # ========================================================================
    # PHASE 2: Model Selection & Data Extraction
    # ========================================================================
    log("=" * 60)
    log("PHASE 2: Model Selection & Data Extraction")
    log("=" * 60)

    # Load reasoning model on CPU to save GPU memory
    phase2_start = time.time()
    model_loader = ModelLoader(
        model_name="roneneldan/TinyStories-33M",
        device=model_device
    )
    model_loader.load_model()
    weights_reasoning = model_loader.extract_weights()
    log(f"✓ Reasoning model loaded on CPU and weights extracted ({time.time() - phase2_start:.1f}s)")

    # Create control model (optional)
    weights_control = None
    if INCLUDE_CONTROL_MODEL:
        log("Creating control model (random initialization)...")
        control_start = time.time()
        weights_control = model_loader.create_random_model()
        log(f"✓ Control model created ({time.time() - control_start:.1f}s)")
    else:
        log("⏭ Skipping control model (INCLUDE_CONTROL_MODEL=False)")

    # ========================================================================
    # PHASE 3: Spectral Computation
    # ========================================================================
    log("=" * 60)
    log("PHASE 3: Spectral Computation")
    log("=" * 60)

    spectral_analyzer = SpectralAnalyzer(device=svd_device)

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

        log(f"Computing circuits for layer {layer_idx}...")

        # Compute QK circuit
        circuit_start = time.time()
        W_QK_Full = circuit_analyzer.compute_qk_circuit(
            W_E, head_weights['W_Q'], head_weights['W_K']
        )
        all_qk_circuits.append(W_QK_Full)
        log(f"✓ QK circuit computed: shape {W_QK_Full.shape} ({time.time() - circuit_start:.1f}s)")

        # Compute OV circuit
        circuit_start = time.time()
        W_OV_Full = circuit_analyzer.compute_ov_circuit(
            W_E, head_weights['W_O'], head_weights['W_V'], W_U
        )
        all_ov_circuits.append(W_OV_Full)
        log(f"✓ OV circuit computed: shape {W_OV_Full.shape} ({time.time() - circuit_start:.1f}s)")

        # Eigen-decomposition for QK
        log(">>> Starting QK eigen-decomposition (this takes ~20-40 min)...")
        svd_start = time.time()
        eigenvals_qk, eigenvecs_qk = spectral_analyzer.compute_eigen_decomposition(W_QK_Full)
        qk_time = time.time() - svd_start
        log(f"✓✓✓ QK SVD COMPLETE! {len(eigenvals_qk)} eigenvalues ({qk_time:.1f}s = {qk_time/60:.1f} min)")

        # Eigen-decomposition for OV
        log(">>> Starting OV eigen-decomposition (this takes ~20-40 min)...")
        svd_start = time.time()
        eigenvals_ov, eigenvecs_ov = spectral_analyzer.compute_eigen_decomposition(W_OV_Full)
        ov_time = time.time() - svd_start
        log(f"✓✓✓ OV SVD COMPLETE! {len(eigenvals_ov)} eigenvalues ({ov_time:.1f}s = {ov_time/60:.1f} min)")

        all_eigenvalues_qk.append(eigenvals_qk)
        all_eigenvalues_ov.append(eigenvals_ov)
        all_eigenvectors_qk.append(eigenvecs_qk)
        all_eigenvectors_ov.append(eigenvecs_ov)

        # Analyze distribution
        qk_stats = spectral_analyzer.analyze_eigenvalue_distribution(eigenvals_qk)
        ov_stats = spectral_analyzer.analyze_eigenvalue_distribution(eigenvals_ov)

        log(f"QK Circuit Statistics:")
        log(f"  Mean: {qk_stats['mean']:.6f}")
        log(f"  Std: {qk_stats['std']:.6f}")
        log(f"  Near zero ratio: {qk_stats.get('near_zero_ratio', 0):.4f}")

        log(f"OV Circuit Statistics:")
        log(f"  Mean: {ov_stats['mean']:.6f}")
        log(f"  Std: {ov_stats['std']:.6f}")
        log(f"  Near zero ratio: {ov_stats.get('near_zero_ratio', 0):.4f}")

        # Plot distributions
        log("Generating plots...")
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
        log("✓ Plots saved to results/")

    # ========================================================================
    # PHASE 4: Comparative Analysis (only if control model enabled)
    # ========================================================================
    if INCLUDE_CONTROL_MODEL and weights_control is not None:
        log("=" * 60)
        log("PHASE 4: Comparative Analysis")
        log("=" * 60)

        comparative_analyzer = ComparativeAnalyzer(device=device)

        # Compute control model circuits
        W_E_control = weights_control['W_E']
        W_U_control = weights_control['W_U']

        if 'attention_weights' in weights_control and len(weights_control['attention_weights']) > 0:
            layer_idx = list(weights_control['attention_weights'].keys())[0]
            head_weights_control = weights_control['attention_weights'][layer_idx]

            log(f"Computing control model circuits for layer {layer_idx}...")

            circuit_start = time.time()
            W_QK_Full_control = circuit_analyzer.compute_qk_circuit(
                W_E_control, head_weights_control['W_Q'], head_weights_control['W_K']
            )
            log(f"✓ Control QK circuit computed ({time.time() - circuit_start:.1f}s)")

            circuit_start = time.time()
            W_OV_Full_control = circuit_analyzer.compute_ov_circuit(
                W_E_control, head_weights_control['W_O'], head_weights_control['W_V'], W_U_control
            )
            log(f"✓ Control OV circuit computed ({time.time() - circuit_start:.1f}s)")

            # Eigen-decomposition for control QK
            log(">>> Starting Control QK eigen-decomposition...")
            svd_start = time.time()
            eigenvals_qk_control, _ = spectral_analyzer.compute_eigen_decomposition(W_QK_Full_control)
            log(f"✓✓✓ Control QK SVD COMPLETE! ({time.time() - svd_start:.1f}s = {(time.time() - svd_start)/60:.1f} min)")

            # Eigen-decomposition for control OV
            log(">>> Starting Control OV eigen-decomposition...")
            svd_start = time.time()
            eigenvals_ov_control, _ = spectral_analyzer.compute_eigen_decomposition(W_OV_Full_control)
            log(f"✓✓✓ Control OV SVD COMPLETE! ({time.time() - svd_start:.1f}s = {(time.time() - svd_start)/60:.1f} min)")

            # Compare
            comparison_qk = comparative_analyzer.compare_spectral_densities(
                all_eigenvalues_qk[0],
                eigenvals_qk_control,
                labels=("Reasoning Model", "Control Model")
            )

            log(f"QK Circuit Comparison:")
            log(f"  Mean difference: {comparison_qk['mean_difference']:.6f}")
            log(f"  Near zero ratio difference: {comparison_qk.get('near_zero_ratio_diff', 0):.4f}")

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
            log(f"Rank Analysis:")
            log(f"  Mean rank ratio: {rank_results['mean_rank_ratio']:.4f}")
            log(f"  Is high rank: {rank_results['is_high_rank']}")
    else:
        log("=" * 60)
        log("PHASE 4: Skipped (control model disabled)")
        log("=" * 60)

    # ========================================================================
    # ANALYSIS COMPLETE
    # ========================================================================
    total_time = time.time() - total_start
    log("=" * 60)
    log("ANALYSIS COMPLETE!")
    log("=" * 60)
    log(f"Total runtime: {total_time:.1f}s = {total_time/60:.1f} min")
    log(f"Results saved to 'results/' directory")
    log("Conclusion: Eigen-analysis provides insights into:")
    log("  - Where attention patterns form (QK circuit)")
    log("  - What information is moved (OV circuit)")
    if INCLUDE_CONTROL_MODEL:
        log("  - Spectral signatures of reasoning vs non-reasoning models")


if __name__ == "__main__":
    main()
