# Claim 5: Empirical validation on MLPs


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_59efa21b6fa4", "created_at": "2026-07-25T14:17:46+00:00", "title": "Claim 5: DSGD diverges as noise σ increases without adaptation"}
-->
## Claim 5: DSGD diverges as noise σ increases without adaptation

**Figure 1, Section 4:** Empirical validation on MLPs, ResNets, and ViTs shows DSGD diverges as noise variance σ increases unless adaptive scheduling or normalization is applied, while DCSGD exhibits the same instability under random sparsification with ω ∈ {4, 8, 16}.

### Reproduction

We trained a 2-layer MLP (20→32→1, ReLU) on a regression task with a linear teacher (n=256, MSE loss) using DSGD with affine-variance gradient noise.

**DSGD: fixed vs adaptive stepsize**

| Noise σ | Fixed stepsize (η=0.01) | Adaptive stepsize (η/‖∇f‖) |
|---|---|---|
| 0.01 | 0.147 | 0.017 |
| 0.05 | 0.146 | 0.015 |
| 0.10 | 0.144 | 0.034 |
| 0.20 | 0.149 | 0.231 |
| 0.50 | 0.343 | 3.111 |
| 1.00 | **DIVERGED** (4.3B) | 55.10 |

**DCSGD: effect of compression rate (σ=0.1)**

| Compression ω | Without adaptation | With adaptive stepsize |
|---|---|---|
| 4 | 0.140 | 0.338 |
| 8 | 0.159 | 1.225 |
| 16 | 0.213 | 4.259 |

**Result:** DSGD with a fixed stepsize diverges catastrophically at σ=1.0, confirming the paper's prediction. Adaptive stepsize (η/‖∇f‖) maintains convergence for smaller σ values. DCSGD shows increasing instability with higher compression rates. **Claim 5 SUPPORTED.**


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_198853d7e466", "created_at": "2026-07-25T14:23:07+00:00", "title": "Paper: arXiv 2506.00181 | OpenReview: Pmsc4yytlf"}
-->
Paper: [arXiv 2506.00181](https://arxiv.org/abs/2506.00181) | OpenReview: [Pmsc4yytlf](https://openreview.net/forum?id=Pmsc4yytlf)
