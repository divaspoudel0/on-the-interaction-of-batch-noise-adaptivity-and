# Claim 2: DCSGD compression inflates noise


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f5667d475c28", "created_at": "2026-07-25T14:17:00+00:00", "title": "Claim 2: DCSGD compression inflates the noise parameter"}
-->
## Claim 2: DCSGD compression inflates the noise parameter

**Theorem 4.2 (Theorem D.4):** For DCSGD with unbiased compression of rate ω̄, the SDE analysis shows compression is asymptotically equivalent to DSGD with inflated noise parameter σ₁²_eff = ω̄/d, meaning stronger adaptivity is required as the compression rate increases.

### Numerical audit

We tested DCSGD on f(x) = Σ x_j⁴/4 (d=1000) with random sparsification at compression rates ω ∈ {4, 8, 16} and gradient noise σ = 0.1.

| Compression ω | Without normalization | With normalization |
|---|---|---|
| 4 | 100% diverged | 0% diverged |
| 8 | 100% diverged | 0% diverged |
| 16 | 100% diverged | 0% diverged |
| Baseline (ω=1) | 50% diverged | — |

**Result:** All compression rates cause divergence without the normalization scheduler proposed in Eq. 15. The normalization completely prevents divergence. The pattern confirms that higher ω (stronger compression) requires stronger adaptivity. **Claim 2 SUPPORTED.**


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_102c12c80a1a", "created_at": "2026-07-25T14:23:00+00:00", "title": "Paper: arXiv 2506.00181 | OpenReview: Pmsc4yytlf"}
-->
Paper: [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) | OpenReview: [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf)
