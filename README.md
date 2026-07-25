# Reproduction: On the Interaction of Batch Noise, Adaptivity, and Compression, under (L₀,L₁)-Smoothness

**Paper:** [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) (ICML 2026, OpenReview [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf))

**Authors:** Enea Monzio Compagnoni, Rustem Islamov, Frank Norbert Proske, Aurelien Lucchi, Antonio Orvieto, Eduard Gorbunov

**Trackio Logbook:** [Hugging Face Space](https://huggingface.co/spaces/divaspoudel/repro-on-the-interaction-of-batch-noise-adaptivity-and-compression-under-l-0-l-1-smoothness-an-s)

## Overview

This repository numerically verifies 5 key claims from the paper, which establishes SDE-based convergence analyses for SGD variants under (L₀,L₁)-smoothness with unbounded-variance gradient noise.

## Claims Verified

| # | Claim | Theorem | Verdict |
|---|-------|---------|---------|
| 1 | DSGD requires stepsize η·η_t < bound from Theorem D.2 | Thm 4.1 (D.2) | **Supported** |
| 2 | DCSGD with random sparsification inflates noise by ω/d | Thm 4.2 (D.4) | **Supported** |
| 3 | Gradient normalization is a necessary stability condition when σ₁² > 0 | §4.1 | **Supported** |
| 4 | DSignSGD converges under Student-t ν=1 noise with constant stepsize | Thm 4.4 (D.9) | **Supported** |
| 5 | MLP diverges with fixed stepsize under high noise; adaptive stabilizes | §4, Fig 1 | **Supported** |

## Quick Start

### Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/divaspoudel0/on-the-interaction-of-batch-noise-adaptivity-and/blob/main/reproduction.ipynb)

### Local

```bash
pip install numpy matplotlib
python experiments/claim1_stepsize.py    # Claim 1
python experiments/claim2_compression.py # Claim 2
python experiments/claim3_normalization.py # Claim 3
python experiments/claim4_dsignsgd.py    # Claim 4
python experiments/claim5_mlp.py         # Claim 5
python experiments/run_all.py            # Run all
python plots/generate_figures.py         # Generate all figures
```

## Project Structure

```
src/
  objectives/
    quartic.py     # f(x) = Σ x_j⁴/4  (L₀=0, L₁=3)
    quadratic.py   # f(x) = ½ xᵀAx    (condition number κ)
  optimizers/
    sgd.py         # DSGD (adaptive/decay/noise options)
    dcsgd.py       # DCSGD (random sparsification compression)
    dsignsgd.py    # DSignSGD (sign-based + noise)
    noise.py       # AffineGaussianNoise, StudentTNoise
    mlp.py         # 2-layer ReLU MLP for Claim 5
experiments/
  claim1_stepsize.py
  claim2_compression.py
  claim3_normalization.py
  claim4_dsignsgd.py
  claim5_mlp.py
  run_all.py
plots/
  generate_figures.py
  claim1_stepsize.png
  claim2_compression.png
  claim3_normalization.png
  claim4_dsignsgd.png
  claim5_mlp.png
reproduction.ipynb   # Colab-ready notebook
```

## Key Experiments

### Claim 1: Stepsize Bound (Quartic)
- Objective: f(x) = Σ x_j⁴/4 on d=100, N=8
- Stepsizes within theoretical bound (<0.099) converge; beyond diverge

### Claim 2: DCSGD Compression (Quartic)
- Random sparsification with ω ∈ {4, 8, 16}
- Without normalization: 100% divergence. With normalization: 0% divergence

### Claim 3: Gradient Normalization (Quadratic)
- Quadratic with condition number κ=100, σ₁² ∈ {0, 0.01, 0.05, 0.1}
- Normalization eliminates divergence for all σ₁² > 0

### Claim 4: DSignSGD Heavy-Tailed (1D Quartic)
- Student-t ν=1 noise (unbounded mean), σ ∈ {0.25, 0.5, 1, 2, 8, 16}
- DSignSGD+scheduler converges (0% div); DSGD diverges (97-100%)

### Claim 5: MLP (2-layer ReLU)
- Architecture: R²⁰ → ReLU(32) → R, MSE loss with linear teacher
- DSGD diverges at σ=1.0 with fixed stepsize; adaptive stepsize converges

## Reproducibility

- **Python:** 3.10+ (tested with 3.14)
- **Dependencies:** numpy, matplotlib (only)
- **GPU:** Not required (CPU-only experiments)
- **Runtime:** ~5 minutes total for all experiments
- **Random seeds:** All experiments use fixed seeds for reproducibility

## License

This is an independent reproduction. The original paper's code has not been released.

## Citation

```bibtex
@article{monzio2025interaction,
  title={On the Interaction of Batch Noise, Adaptivity, and Compression, under (L_0,L_1)-Smoothness: An SDE Approach},
  author={Monzio Compagnoni, Enea and Islamov, Rustem and Proske, Frank Norbert and Lucchi, Aurelien and Orvieto, Antonio and Gorbunov, Eduard},
  journal={arXiv preprint arXiv:2506.00181},
  year={2025}
}
```
