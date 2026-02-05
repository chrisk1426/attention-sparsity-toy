# Spectral Analysis of Toy Models

This project implements spectral analysis of transformer attention circuits to understand where and how "thinking" happens in language models.

## Project Overview

Based on the "Mathematical Framework for Transformers" (Elhage et al., 2021), this project analyzes:
- **QK Circuit (The "Where")**: Determines attention patterns
- **OV Circuit (The "What")**: Determines what information is moved

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `circuit_analysis.py`: Core implementation of QK and OV circuits
- `model_loader.py`: Model loading and weight extraction
- `spectral_computation.py`: Eigenvalue/eigenvector computation
- `comparative_analysis.py`: Comparative analysis between models
- `bias_detection.py`: Bias probing using eigenvectors
- `main.py`: Main execution script

## Usage

```bash
python main.py
```

## Phases

1. **Phase 1**: Theoretical Framework & Definitions
2. **Phase 2**: Model Selection & Data Extraction
3. **Phase 3**: Spectral Computation
4. **Phase 4**: Comparative Analysis
5. **Phase 5**: Bias Detection & Conclusion
