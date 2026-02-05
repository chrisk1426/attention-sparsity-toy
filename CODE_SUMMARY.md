
# Code Summary: Spectral Analysis of Transformer Attention Circuits

## Overview

This project analyzes transformer attention mechanisms using spectral (eigenvalue) analysis to understand how language models process information. It implements a 4-phase analysis pipeline that decomposes attention circuits into their fundamental mathematical components and compares reasoning-capable models against control models.

**Main Goal**: Understand where and how "thinking" happens in transformer models by analyzing the spectral properties of attention circuits.

---

## Core Concepts

### QK Circuit (The "Where")
- **Formula**: `W_QK_Full = W_E^T * W_QK_head * W_E`
- **Purpose**: Determines attention patterns - where information flows in the model
- **Matrix Size**: `[vocab_size, vocab_size]` - maps token-to-token attention scores

### OV Circuit (The "What")
- **Formula**: `W_OV_Full = W_U * W_OV_head * W_E`
- **Purpose**: Determines what information is moved and how it affects final outputs
- **Matrix Size**: `[vocab_size, vocab_size]` - maps token-to-token information transfer

---

## Project Structure

### Main Modules

#### 1. `main.py` - Execution Orchestrator
- **Role**: Coordinates all phases of the analysis
- **Flow**:
  1. Initializes analyzers and loads models
  2. Extracts weights from reasoning model (TinyStories-33M)
  3. Creates control model (random initialization)
  4. Computes circuits and performs spectral analysis
  5. Compares reasoning vs control models
  6. Generates plots and saves results

#### 2. `model_loader.py` - Model & Weight Extraction
- **Class**: `ModelLoader`
- **Key Methods**:
  - `load_model()`: Loads transformer model from HuggingFace
  - `extract_weights()`: Extracts embedding (W_E), unembedding (W_U), and attention head weights (W_Q, W_K, W_V, W_O)
  - `create_random_model()`: Creates randomly initialized control model with same architecture
- **Handles**: Multiple model architectures (GPT-style, different naming conventions)

#### 3. `circuit_analysis.py` - Circuit Computation
- **Class**: `CircuitAnalyzer`
- **Key Methods**:
  - `compute_qk_circuit()`: Computes full QK circuit matrix
  - `compute_ov_circuit()`: Computes full OV circuit matrix
  - `compute_circuits_for_head()`: Computes both circuits for a single head
- **Output**: Full vocab×vocab matrices representing attention circuits

#### 4. `spectral_computation.py` - Eigenvalue Analysis
- **Class**: `SpectralAnalyzer`
- **Key Methods**:
  - `compute_eigen_decomposition()`: Performs eigenvalue/eigenvector decomposition
    - Uses `np.linalg.eigh()` for symmetric matrices
    - Uses `np.linalg.svd()` for non-symmetric matrices
  - `analyze_eigenvalue_distribution()`: Computes statistics (mean, std, near-zero ratio)
  - `plot_eigenvalue_distribution()`: Generates histograms of eigenvalue distributions
  - `compute_matrix_rank()`: Calculates effective rank using SVD
- **Focus**: Analyzes tail distribution near zero (sparsity signature)

#### 5. `comparative_analysis.py` - Model Comparison
- **Class**: `ComparativeAnalyzer`
- **Key Methods**:
  - `compare_spectral_densities()`: Compares eigenvalue distributions between models
  - `test_rank_hypothesis()`: Tests if matrices are high-rank (null hypothesis)
  - `plot_comparison()`: Side-by-side comparison plots
  - `identify_thinking_heads()`: Identifies heads with unique spectral signatures
- **Purpose**: Quantify differences between reasoning and non-reasoning models

---

## Execution Flow

### Phase 1: Theoretical Framework & Definitions
```
1. Initialize CircuitAnalyzer
2. Print circuit formulas and definitions
```

### Phase 2: Model Selection & Data Extraction
```
1. Load TinyStories-33M model via ModelLoader
2. Extract weights:
   - W_E (embedding matrix)
   - W_U (unembedding matrix)
   - Attention head weights per layer (W_Q, W_K, W_V, W_O)
3. Create control model (random initialization)
```

### Phase 3: Spectral Computation
```
For each layer/head:
1. Compute QK circuit: W_QK_Full = W_E^T * (W_Q * W_K^T) * W_E
2. Compute OV circuit: W_OV_Full = W_U * (W_V * W_O^T) * W_E^T
3. Eigen-decomposition of both circuits
4. Analyze eigenvalue distribution:
   - Mean, std, median
   - Near-zero ratio (sparsity measure)
5. Generate plots:
   - Full eigenvalue distribution
   - Zoomed view near zero (tail distribution)
```

### Phase 4: Comparative Analysis
```
1. Compute same circuits for control model
2. Compare spectral densities:
   - Mean difference
   - Near-zero ratio difference
3. Rank analysis:
   - Compute effective rank of all circuits
   - Test high-rank hypothesis
4. Generate comparison plots
```

---

## Data Flow

```
Model Weights (state_dict)
    ↓
ModelLoader.extract_weights()
    ↓
{ W_E, W_U, attention_weights[layer][head] }
    ↓
CircuitAnalyzer.compute_qk_circuit() / compute_ov_circuit()
    ↓
Full Circuit Matrices [vocab_size × vocab_size]
    ↓
SpectralAnalyzer.compute_eigen_decomposition()
    ↓
Eigenvalues + Eigenvectors
    ↓
SpectralAnalyzer.analyze_eigenvalue_distribution()
    ↓
Statistics + Plots
    ↓
ComparativeAnalyzer.compare_spectral_densities()
    ↓
Comparison Results + Visualizations
```

---

## Key Features

### 1. Multi-Architecture Support
- Handles different transformer architectures (GPT-style, various naming conventions)
- Automatically detects embedding/unembedding layers
- Flexible attention weight extraction

### 2. Spectral Analysis Focus
- Emphasizes eigenvalue distributions, especially near zero
- Computes sparsity metrics (near-zero ratio)
- Visualizes tail distributions

### 3. Control Experiments
- Creates randomly initialized control models
- Enables comparison between trained and untrained models
- Tests hypotheses about learned vs. random patterns

### 4. Visualization
- Eigenvalue distribution histograms
- Side-by-side model comparisons
- Focus on zero-region (sparsity signature)

---

## Output Files

All results are saved to `results/` directory:
- `qk_eigenvalue_distribution.png`: QK circuit eigenvalue distribution
- `ov_eigenvalue_distribution.png`: OV circuit eigenvalue distribution
- `comparison_qk.png`: Side-by-side comparison of reasoning vs control models

---

## Dependencies

- **torch**: Deep learning framework
- **transformers**: HuggingFace model loading
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **matplotlib**: Plotting
- **seaborn**: Statistical visualizations

---

## Current Limitations / Known Issues

1. **Variable Scope**: `W_QK_Full` and `W_OV_Full` are defined inside conditional blocks but used outside (lines 192) - will cause `NameError` if attention weights aren't found
2. **Index Access**: `all_eigenvalues_qk[0]` accessed without checking if list is empty (lines 173, 184) - will cause `IndexError` if no eigenvalues computed
3. **Inefficiency**: `model_loader.py` calls `extract_weights()` multiple times unnecessarily

---

## Research Questions Addressed

1. **Do reasoning models show different spectral signatures?**
   - Answered via `compare_spectral_densities()`

2. **Are attention circuits high-rank or low-rank?**
   - Answered via `test_rank_hypothesis()`

3. **What is the sparsity pattern (near-zero eigenvalues)?**
   - Answered via `analyze_eigenvalue_distribution()` with focus on zero region

4. **Which heads have unique spectral signatures?**
   - Answered via `identify_thinking_heads()` (if extended to multiple heads)

---

## Mathematical Foundation

Based on the "Mathematical Framework for Transformers" (Elhage et al., 2021):
- Attention can be decomposed into QK (where) and OV (what) circuits
- Full circuits are vocab×vocab matrices that capture token-to-token relationships
- Spectral analysis reveals the structure and sparsity of these relationships
- Eigenvalues near zero indicate sparse, selective attention patterns
