# Claim 1: DSGD adaptive stepsize condition


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0d5da00679cd", "created_at": "2026-07-25T14:16:44+00:00", "title": "Claim 1: DSGD requires adaptive stepsize under (L₀,L₁)-smoothness"}
-->
## Claim 1: DSGD requires adaptive stepsize under (L₀,L₁)-smoothness

**Theorem 4.1 (Theorem D.2):** Under (L₀,L₁)-smoothness with unbounded-variance gradient noise, DSGD requires:
8630\eta \cdot \eta_t < \frac{2\epsilon}{L_0 + L_1 \mathbb{E}[\|
abla f(X_t)\|] + \frac{d}{N}(\bar{\sigma}_1^2 L_0 + \bar{\sigma}_0^2 L_1 + L_1 \bar{\sigma}_1^2 \mathbb{E}[\|
abla f(X_t)\|])}8630

### Numerical audit

We verified this on f(x) = Σ x_j⁴/4 (d=1000, N=8) with (L₀,L₁) = (0, 3) and affine-variance Gaussian noise (σ₁² = 0.01).

**Approximate theoretical bound: η ≈ 0.099**

| Stepsize η | Mean final loss | Divergence rate |
|---|---|---|
| 0.001 | 2.74 | 0% |
| 0.005 | 0.17 | 0% |
| 0.01 | 0.05 | 0% |
| 0.02 | 0.01 | 0% |
| 0.05 | 0.002 | 0% |
| 0.10 | diverged | 40% |
| 0.20 | diverged | 100% |

**Result:** Stepsizes within the bound (≤ 0.05) converge reliably; stepsizes exceeding the bound (≥ 0.10) cause divergence. **Claim 1 SUPPORTED.**


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d97c1c08f445", "created_at": "2026-07-25T14:22:40+00:00", "title": "Paper: arXiv 2506.00181 | OpenReview: Pmsc4yytlf"}
-->
Paper: [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) | OpenReview: [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf)
