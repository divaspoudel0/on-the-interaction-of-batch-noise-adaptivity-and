#!/usr/bin/env python3
"""
Experiment 1: DSGD Adaptive Stepsize Condition (Claim 1)

Theorem 4.1 (Theorem D.2): Under (L0,L1)-smoothness with unbounded-variance
gradient noise, DSGD requires:

  eta * eta_t < 2*epsilon / [L0 + L1*E[||grad||] + (d/N)*(sigma1^2*L0 + sigma0^2*L1 + L1*sigma1^2*E[||grad||])]

We verify this on f(x) = sum(x_j^4)/4 (quartic) with (L0,L1) = (0, 3),
showing stepsizes within the bound converge while those exceeding it diverge.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from objectives.quartic import QuarticObjective
from optimizers.sgd import DSGD
from optimizers.noise import AffineGaussianNoise
from utils import save_results


def compute_stepsize_bound(d, N, L0, L1, sigma0_sq, sigma1_sq, grad_norm, epsilon=0.5):
    """Compute the theoretical stepsize bound from Theorem D.2."""
    return 2 * epsilon / (
        L0 + L1 * grad_norm
        + (d / N) * (sigma1_sq * L0 + sigma0_sq * L1 + L1 * sigma1_sq * grad_norm)
    )


def run_experiment():
    print("=" * 60)
    print("Experiment 1: DSGD Adaptive Stepsize Condition (Claim 1)")
    print("=" * 60)

    # Paper parameters
    d = 100
    N = 8
    L0, L1 = 0.0, 3.0
    sigma0_sq, sigma1_sq = 0.0, 0.01
    epsilon = 0.5

    # Compute theoretical bound
    typical_grad_norm = 3.0
    bound = compute_stepsize_bound(d, N, L0, L1, sigma0_sq, sigma1_sq, typical_grad_norm, epsilon)
    print(f"\nTheoretical stepsize bound: {bound:.6f}")

    # Test multiple stepsizes
    stepsizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    n_steps = 1000
    n_runs = 30

    results = {"theoretical_bound": float(bound), "stepsizes": {}}

    for eta in stepsizes:
        noise = AffineGaussianNoise(sigma0_sq, sigma1_sq)
        opt = DSGD(lr=eta, noise=noise)
        obj = QuarticObjective(d)
        x0 = np.random.randn(d) * 1.5

        stats = opt.run(obj, x0, n_steps=n_steps, n_runs=n_runs, seed=42)
        results["stepsizes"][str(eta)] = {
            "mean_final_loss": stats["mean_loss"],
            "divergence_rate": stats["divergence_rate"],
            "within_bound": eta < bound,
        }
        status = "CONVERGE" if stats["divergence_rate"] < 0.5 else "DIVERGE"
        bound_status = "within" if eta < bound else "beyond"
        print(f"  eta={eta:.4f} ({bound_status} bound): "
              f"mean_loss={stats['mean_loss']:.6f}, div={stats['divergence_rate']:.0%} [{status}]")

    # Verify claim
    within = [e for e in stepsizes if e < bound]
    beyond = [e for e in stepsizes if e > bound]
    converges_within = all(results["stepsizes"][str(e)]["divergence_rate"] < 0.5 for e in within)
    diverges_beyond = any(results["stepsizes"][str(e)]["divergence_rate"] > 0.5 for e in beyond)
    results["claim_supported"] = converges_within and diverges_beyond

    print(f"\nClaim 1: Stepsizes within bound converge: {converges_within}")
    print(f"Claim 1: Stepsizes beyond bound diverge: {diverges_beyond}")
    print(f"Claim 1 SUPPORTED: {results['claim_supported']}")

    save_results(results, "experiments/outputs/claim1_stepsize_results.json")
    return results


if __name__ == "__main__":
    run_experiment()
