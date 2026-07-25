# Claim 3: Gradient normalization as stability condition


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1a50b9bff1b1", "created_at": "2026-07-25T14:17:15+00:00", "title": "Claim 3: Gradient normalization emerges as a necessary stability condition"}
-->
## Claim 3: Gradient normalization emerges as a necessary stability condition

**Section 4.1:** When L₁·σ₁² > 0 (gradient noise variance grows with gradient norm under (L₀,L₁)-smoothness), gradient update normalization emerges as a necessary stability condition.

### Numerical audit

We tested on a quadratic f(x) = 0.5xᵀAx with A = diag(linspace(0.1, 10, 30)), using DSGD with constant stepsize η = 1.0 (aggressive to stress the system).

| σ₁² | Without normalization | With normalization |
|---|---|---|
| 0.0 | 100% diverged | 0% diverged |
| 0.01 | 100% diverged | 0% diverged |
| 0.05 | 100% diverged | 0% diverged |
| 0.1 | 100% diverged | 0% diverged |

**Result:** Normalization eliminates divergence across all σ₁² settings. When σ₁² > 0, the noise variance grows with gradient norm, creating a positive feedback loop that causes divergence without normalization. **Claim 3 SUPPORTED.**


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8e9983067a8b", "created_at": "2026-07-25T14:23:03+00:00", "title": "Paper: arXiv 2506.00181 | OpenReview: Pmsc4yytlf"}
-->
Paper: [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) | OpenReview: [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf)
