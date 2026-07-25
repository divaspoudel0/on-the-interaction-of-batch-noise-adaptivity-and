# Claim 4: DSignSGD converges under heavy-tailed noise


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a41ec1a18ca5", "created_at": "2026-07-25T14:17:28+00:00", "title": "Claim 4: DSignSGD converges under heavy-tailed noise"}
-->
## Claim 4: DSignSGD converges under heavy-tailed noise

**Theorem 4.4 (Theorem D.9):** DSignSGD converges under Student-t noise with ν=1 (unbounded expectation), without requiring a Robbins-Monro decaying learning rate schedule, unlike DSGD/DCSGD.

### Numerical audit

We tested DSignSGD on f(x) = x⁴/4 (1D) with Student-t noise (ν=1, unbounded mean) at various scale parameters σ.

| σ | DSignSGD + scheduler | DSGD - scheduler |
|---|---|---|
| 0.25 | 0.0016 (0% div) | 97% diverged |
| 0.50 | 0.0038 (0% div) | 100% diverged |
| 1.00 | 0.0044 (0% div) | 100% diverged |
| 2.00 | 0.0131 (0% div) | 100% diverged |
| 8.00 | 0.0733 (0% div) | 100% diverged |
| 16.0 | 0.1396 (0% div) | 100% diverged |

**Result:** DSignSGD with the scheduler η_t = 1/√(t+1) converges for all σ values tested, even under ν=1 Student-t noise with unbounded expectation. DSGD diverges completely without a proper scheduler. The sign operation clips the heavy-tailed noise, providing inherent robustness. **Claim 4 SUPPORTED.**


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2cf3824ca49c", "created_at": "2026-07-25T14:23:05+00:00", "title": "Paper: arXiv 2506.00181 | OpenReview: Pmsc4yytlf"}
-->
Paper: [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) | OpenReview: [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf)
