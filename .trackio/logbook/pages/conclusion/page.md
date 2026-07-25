# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_55a5b6222507", "created_at": "2026-07-25T14:18:00+00:00", "title": "Conclusion"}
-->
## Conclusion

### Overall findings

All 5 major claims of the paper were **supported** by our reproduction:

1. **Claim 1 SUPPORTED:** DSGD requires adaptive stepsize under (L₀,L₁)-smoothness. Numerical audit on quartic objective confirmed stepsizes within the theoretical bound converge while those exceeding it diverge.

2. **Claim 2 SUPPORTED:** DCSGD compression inflates the noise parameter. Random sparsification with ω ∈ {4, 8, 16} causes divergence without the normalization scheduler; all rates converge with normalization.

3. **Claim 3 SUPPORTED:** Gradient normalization is a necessary stability condition when L₁·σ₁² > 0. Quadratic objective experiments confirm normalization eliminates divergence.

4. **Claim 4 SUPPORTED:** DSignSGD converges under Student-t noise with ν=1 (unbounded expectation). All tested noise scales converge with η_t = 1/√(t+1), while DSGD diverges without proper scheduling.

5. **Claim 5 SUPPORTED:** DSGD diverges as noise σ increases without adaptive scheduling. MLP experiments show fixed stepsize diverges at σ=1.0 while adaptive methods maintain convergence.

### Reproducibility notes

- **No official code available** (arXiv 2506.00181, authors confirm Code: TBD). All implementations are independent.
- Experiments used **toy-scale problems** (quartic/quadratic objectives, small MLP) since full-scale ResNet-18/ViT experiments require distributed GPU infrastructure not available for this reproduction.
- The **theoretical results are the primary contribution** of this paper; our numerical audits validate the key inequalities and stability conditions.
- All experiments run on CPU in < 60 seconds total, demonstrating the claims can be verified with minimal compute.

### Paper information
- **arXiv:** [2506.00181](https://arxiv.org/abs/2506.00181)
- **Venue:** ICML 2026 (Poster)
- **Authors:** Enea Monzio Compagnoni, Rustem Islamov, Frank Norbert Proske, Aurelien Lucchi, Antonio Orvieto, Eduard Gorbunov
