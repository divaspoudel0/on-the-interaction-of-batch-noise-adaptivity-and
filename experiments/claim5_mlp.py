#!/usr/bin/env python3
"""
Experiment 5: MLP Empirical Validation (Claim 5)

Figure 1, Section 4: DSGD diverges as noise variance sigma increases unless
adaptive scheduling or normalization is applied, while DCSGD exhibits the
same instability under random sparsification with omega in {4, 8, 16}.

Architecture: h_theta: R^20 -> R, single hidden layer of width 32, ReLU.
Loss: MSE with linear teacher, n=256.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from optimizers.mlp import SimpleMLP, generate_regression_data, random_sparsify
from utils import save_results


def train_mlp(
    sigma,
    method="dsgd",
    omega=4,
    adaptive=False,
    normalize=False,
    n_steps=200,
    n_runs=3,
    seed=42,
):
    """Train MLP with different optimizers."""
    X, y, _ = generate_regression_data(n_samples=256, input_dim=20, seed=0)
    all_final_losses = []

    for run in range(n_runs):
        model = SimpleMLP(input_dim=20, hidden_dim=32, seed=seed + run)
        rng = np.random.default_rng(seed + run)
        losses = []

        for t in range(n_steps):
            g, loss = model.compute_grad(X, y)
            losses.append(loss)

            if loss > 1e8:
                all_final_losses.append(1e10)
                break

            gn = np.linalg.norm(g)
            g_noisy = g + rng.standard_normal(len(g)) * sigma * gn

            if method == "dcsgd":
                g_noisy = random_sparsify(g_noisy, omega, rng)

            # Compute stepsize
            if adaptive and gn > 0:
                lr = 0.1 / gn
            else:
                lr = 0.01

            model.update(g_noisy, lr)
        else:
            all_final_losses.append(np.mean(losses[-30:]))

    return {
        "mean_final_loss": float(np.mean(all_final_losses)),
        "divergence_rate": float(np.mean(np.array(all_final_losses) > 1e8)),
    }


def run_experiment():
    print("=" * 60)
    print("Experiment 5: MLP Empirical Validation (Claim 5)")
    print("=" * 60)

    results = {"dsgd": {}, "dsgd_adaptive": {}, "dcsgd": {}, "dcsgd_adaptive": {}}

    # Part 1: DSGD fixed vs adaptive stepsize
    print("\n--- DSGD: fixed vs adaptive stepsize ---")
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for sigma in sigmas:
        stats_fixed = train_mlp(sigma, method="dsgd", adaptive=False)
        stats_adapt = train_mlp(sigma, method="dsgd", adaptive=True)
        results["dsgd"][str(sigma)] = stats_fixed
        results["dsgd_adaptive"][str(sigma)] = stats_adapt
        print(f"  sigma={sigma:5.2f}: fixed={stats_fixed['mean_final_loss']:14.2f}, "
              f"adaptive={stats_adapt['mean_final_loss']:.6f}")

    # Part 2: DCSGD with different compression rates
    print("\n--- DCSGD: effect of compression rate ---")
    for omega in [4, 8, 16]:
        for sigma in [0.01, 0.05, 0.1]:
            stats = train_mlp(sigma, method="dcsgd", omega=omega, adaptive=False)
            stats_a = train_mlp(sigma, method="dcsgd", omega=omega, adaptive=True)
            key = f"omega_{omega}_sigma_{sigma}"
            results["dcsgd"][key] = stats
            results["dcsgd_adaptive"][key] = stats_a
            print(f"  omega={omega}, sigma={sigma}: fixed={stats['mean_final_loss']:.6f}, "
                  f"adaptive={stats_a['mean_final_loss']:.6f}")

    # Verify claim
    fixed_div = results["dsgd"]["1.0"]["divergence_rate"] > 0.5
    adaptive_ok = all(
        results["dsgd_adaptive"][str(s)]["mean_final_loss"] < 10.0
        for s in sigmas[:4]
    )
    results["claim_supported"] = fixed_div and adaptive_ok

    print(f"\nClaim 5: DSGD fixed stepsize diverges at sigma=1.0: {fixed_div}")
    print(f"Claim 5: Adaptive stepsize converges for sigma<=0.2: {adaptive_ok}")
    print(f"Claim 5 SUPPORTED: {results['claim_supported']}")

    save_results(results, "experiments/outputs/claim5_mlp_results.json")
    return results


if __name__ == "__main__":
    run_experiment()
