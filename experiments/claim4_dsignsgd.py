#!/usr/bin/env python3
"""
Experiment 4: DSignSGD Converges Under Heavy-Tailed Noise (Claim 4)

Theorem 4.4 (Theorem D.9): DSignSGD converges under Student-t noise with
nu=1 (unbounded expectation), without requiring a Robbins-Monro decaying
learning rate schedule, unlike DSGD/DCSGD.

We verify on f(x) = x^4/4 (1D) with Student-t noise.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from optimizers.noise import StudentTNoise
from utils import save_results


def run_dsign_1d(sigma, nu, n_steps=800, n_runs=100, use_sched=False, seed=42):
    """Run DSignSGD on 1D quartic."""
    rng = np.random.default_rng(seed)
    noise = StudentTNoise(nu=nu, scale=sigma)
    all_losses = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        x = rng.standard_normal() * 3.0
        for t in range(n_steps):
            grad = x**3
            g = grad + noise(np.array([grad]), rng)[0]
            sign_g = np.sign(g)
            lr = 1.0 / np.sqrt(t + 1) if use_sched else 0.1
            x = x - lr * sign_g
            all_losses[run, t] = x**4 / 4
            if abs(x) > 1e6:
                all_losses[run, t:] = 1e10
                break

    return all_losses


def run_dsgd_1d(sigma, nu, n_steps=800, n_runs=100, use_sched=False, seed=42):
    """Run DSGD on 1D quartic (for comparison)."""
    rng = np.random.default_rng(seed)
    noise = StudentTNoise(nu=nu, scale=sigma)
    all_losses = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        x = rng.standard_normal() * 3.0
        for t in range(n_steps):
            grad = x**3
            g = grad + noise(np.array([grad]), rng)[0]
            lr = 1.0 / (t + 1) if use_sched else 0.1
            x = x - lr * g
            all_losses[run, t] = x**4 / 4
            if abs(x) > 1e6:
                all_losses[run, t:] = 1e10
                break

    return all_losses


def run_experiment():
    print("=" * 60)
    print("Experiment 4: DSignSGD Heavy-Tailed Noise (Claim 4)")
    print("=" * 60)

    nu = 1  # Heavy-tailed (unbounded mean)
    sigmas = [0.25, 0.5, 1.0, 2.0, 8.0, 16.0]
    n_steps = 800
    n_runs = 100

    results = {"nu": nu, "sigma_results": {}}

    for sigma in sigmas:
        # DSignSGD without scheduler
        losses_ds_ns = run_dsign_1d(sigma, nu, n_steps, n_runs, use_sched=False)
        # DSignSGD with scheduler
        losses_ds_s = run_dsign_1d(sigma, nu, n_steps, n_runs, use_sched=True)
        # DSGD without scheduler (should diverge)
        losses_dg_ns = run_dsgd_1d(sigma, nu, n_steps, n_runs, use_sched=False)

        final_ds_ns = losses_ds_ns[:, -100:].mean(axis=1)
        final_ds_s = losses_ds_s[:, -100:].mean(axis=1)
        final_dg_ns = losses_dg_ns[:, -100:].mean(axis=1)

        results["sigma_results"][str(sigma)] = {
            "dsign_no_sched": float(np.mean(final_ds_ns)),
            "dsign_sched": float(np.mean(final_ds_s)),
            "dsgd_no_sched": float(np.mean(final_dg_ns)),
            "dsign_sched_div": float(np.mean(final_ds_s > 1e6)),
        }
        print(f"  sigma={sigma:5.2f}: DSign+sch={np.mean(final_ds_s):.6f} "
              f"(div={np.mean(final_ds_s > 1e6):.0%}), "
              f"DSGD={np.mean(final_dg_ns):.2f} "
              f"(div={np.mean(final_dg_ns > 1e6):.0%})")

    # Verify claim
    all_converge = all(results["sigma_results"][str(s)]["dsign_sched"] < 1.0 for s in sigmas)
    any_diverge = any(results["sigma_results"][str(s)]["dsgd_no_sched"] > 1e5 for s in sigmas[2:])
    results["claim_supported"] = all_converge and any_diverge

    print(f"\nClaim 4: DSignSGD+scheduler converges for all sigma: {all_converge}")
    print(f"Claim 4: DSGD diverges for large sigma: {any_diverge}")
    print(f"Claim 4 SUPPORTED: {results['claim_supported']}")

    save_results(results, "experiments/outputs/claim4_dsignsgd_results.json")
    return results


if __name__ == "__main__":
    run_experiment()
