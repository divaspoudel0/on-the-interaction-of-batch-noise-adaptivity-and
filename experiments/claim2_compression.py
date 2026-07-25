#!/usr/bin/env python3
"""
Experiment 2: DCSGD Compression Inflates Noise (Claim 2)

Theorem 4.2 (Theorem D.4): For DCSGD with unbiased compression of rate omega_bar,
the SDE analysis shows compression is asymptotically equivalent to DSGD with
inflated noise parameter sigma1^2_eff = omega_bar / d.

We verify on f(x) = sum(x_j^4)/4 with random sparsification:
- Without normalization: DCSGD diverges for all omega in {4, 8, 16}
- With normalization: DCSGD always converges
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from objectives.quartic import QuarticObjective
from optimizers.dcsgd import DCSGD
from optimizers.noise import AffineGaussianNoise
from utils import save_results


def run_experiment():
    print("=" * 60)
    print("Experiment 2: DCSGD Compression Inflates Noise (Claim 2)")
    print("=" * 60)

    d = 100
    sigma = 0.1
    omegas = [4, 8, 16]
    n_steps = 1000
    n_runs = 20
    eta = 0.1

    results = {"omega_results": {}}

    # Test DCSGD with different compression rates
    for omega in omegas:
        noise = AffineGaussianNoise(sigma0_sq=0.0, sigma1_sq=sigma**2)
        obj = QuarticObjective(d)
        x0 = np.random.randn(d) * 1.5

        # Without normalization
        opt_no_norm = DCSGD(lr=eta, omega=omega, noise=noise, normalize=False)
        stats_no = opt_no_norm.run(obj, x0, n_steps=n_steps, n_runs=n_runs, seed=42)

        # With normalization
        opt_norm = DCSGD(lr=eta, omega=omega, noise=noise, normalize=True)
        stats_yes = opt_norm.run(obj, x0, n_steps=n_steps, n_runs=n_runs, seed=42)

        results["omega_results"][str(omega)] = {
            "without_normalization": {
                "mean_final_loss": stats_no["mean_loss"],
                "divergence_rate": stats_no["divergence_rate"],
            },
            "with_normalization": {
                "mean_final_loss": stats_yes["mean_loss"],
                "divergence_rate": stats_yes["divergence_rate"],
            },
        }
        print(f"  omega={omega}: no_norm div={stats_no['divergence_rate']:.0%}, "
              f"norm div={stats_yes['divergence_rate']:.0%}")

    # Test baseline (no compression)
    noise = AffineGaussianNoise(sigma0_sq=0.0, sigma1_sq=sigma**2)
    obj = QuarticObjective(d)
    x0 = np.random.randn(d) * 1.5
    opt_base = DCSGD(lr=eta, omega=1.0, noise=noise, normalize=False)
    stats_base = opt_base.run(obj, x0, n_steps=n_steps, n_runs=n_runs, seed=42)
    results["baseline"] = {
        "mean_final_loss": stats_base["mean_loss"],
        "divergence_rate": stats_base["divergence_rate"],
    }
    print(f"  Baseline (omega=1): div={stats_base['divergence_rate']:.0%}")

    # Verify claim: all omegas without norm diverge, all with norm converge
    all_no_norm_diverge = all(
        results["omega_results"][str(w)]["without_normalization"]["divergence_rate"] > 0.5
        for w in omegas
    )
    all_norm_converge = all(
        results["omega_results"][str(w)]["with_normalization"]["divergence_rate"] < 0.5
        for w in omegas
    )
    results["claim_supported"] = all_no_norm_diverge and all_norm_converge

    print(f"\nClaim 2: All compression rates diverge without normalization: {all_no_norm_diverge}")
    print(f"Claim 2: All compression rates converge with normalization: {all_norm_converge}")
    print(f"Claim 2 SUPPORTED: {results['claim_supported']}")

    save_results(results, "experiments/outputs/claim2_compression_results.json")
    return results


if __name__ == "__main__":
    run_experiment()
