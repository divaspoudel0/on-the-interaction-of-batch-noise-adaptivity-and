#!/usr/bin/env python3
"""Generate all publication-quality figures."""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})

PLOTS_DIR = Path(__file__).parent
OUT_DIR = Path(__file__).parent.parent / "experiments" / "outputs"


def load(name):
    with open(OUT_DIR / name) as f:
        return json.load(f)


def plot_claim1():
    """Figure 1: Stepsize condition — final loss vs stepsize with theoretical bound."""
    r = load("claim1_stepsize_results.json")
    bound = r["theoretical_bound"]
    stepsizes = [float(k) for k in r["stepsizes"]]
    losses = [r["stepsizes"][k]["mean_final_loss"] for k in r["stepsizes"]]
    divs = [r["stepsizes"][k]["divergence_rate"] for k in r["stepsizes"]]

    fig, ax = plt.subplots()
    colors = ["#2ecc71" if d < 0.5 else "#e74c3c" for d in divs]
    ax.bar([str(s) for s in stepsizes], losses, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=str(bound), color="navy", linestyle="--", linewidth=2, label=f"Thm bound ≈ {bound:.3f}")
    ax.set_xlabel("Stepsize η")
    ax.set_ylabel("Mean Final Loss")
    ax.set_title("Claim 1: DSGD Stepsize Condition (Quartic, L₀=0, L₁=3)")
    ax.set_yscale("log")
    ax.legend()

    green_patch = plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", ec="black", label="Converge")
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", ec="black", label="Diverge")
    ax.legend(handles=[ax.get_legend_handles_labels()[0][0], green_patch, red_patch],
              labels=[f"Thm bound ≈ {bound:.3f}", "Converge (0% div)", "Diverge (>50% div)"])

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "claim1_stepsize.png")
    plt.close(fig)
    print("  Saved claim1_stepsize.png")


def plot_claim2():
    """Figure 2: DCSGD compression — divergence rates with and without normalization."""
    r = load("claim2_compression_results.json")
    omegas = sorted(r["omega_results"].keys(), key=int)
    no_norm = [r["omega_results"][w]["without_normalization"]["divergence_rate"] * 100 for w in omegas]
    norm = [r["omega_results"][w]["with_normalization"]["divergence_rate"] * 100 for w in omegas]

    x = np.arange(len(omegas))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w/2, no_norm, w, label="Without normalization", color="#e74c3c", edgecolor="black")
    ax.bar(x + w/2, norm, w, label="With normalization", color="#2ecc71", edgecolor="black")
    ax.set_xlabel("Compression Rate ω")
    ax.set_ylabel("Divergence Rate (%)")
    ax.set_title("Claim 2: Compression Inflates Gradient Noise (Quartic)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ω={w}" for w in omegas])
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "claim2_compression.png")
    plt.close(fig)
    print("  Saved claim2_compression.png")


def plot_claim3():
    """Figure 3: Normalization stabilizes across sigma1^2 values."""
    r = load("claim3_normalization_results.json")
    sigmas = sorted(r["sigma1_sq_results"].keys(), key=float)
    no_norm = [r["sigma1_sq_results"][s]["without_normalization"]["divergence_rate"] * 100 for s in sigmas]
    norm = [r["sigma1_sq_results"][s]["with_normalization"]["divergence_rate"] * 100 for s in sigmas]

    x = np.arange(len(sigmas))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w/2, no_norm, w, label="Without normalization", color="#e74c3c", edgecolor="black")
    ax.bar(x + w/2, norm, w, label="With normalization", color="#2ecc71", edgecolor="black")
    ax.set_xlabel("σ₁² (Gradient Noise Scale)")
    ax.set_ylabel("Divergence Rate (%)")
    ax.set_title("Claim 3: Normalization as Stability Condition (Quadratic, κ=100)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ₁²={s}" for s in sigmas])
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "claim3_normalization.png")
    plt.close(fig)
    print("  Saved claim3_normalization.png")


def plot_claim4():
    """Figure 4: DSignSGD under heavy-tailed noise."""
    r = load("claim4_dsignsgd_results.json")
    sigmas = sorted(r["sigma_results"].keys(), key=float)
    ds_vals = [r["sigma_results"][s]["dsign_sched"] for s in sigmas]
    dg_vals = [r["sigma_results"][s]["dsgd_no_sched"] for s in sigmas]

    fig, ax = plt.subplots()
    ax.semilogy([float(s) for s in sigmas], ds_vals, "o-", color="#2ecc71", label="DSignSGD (with scheduler)")
    # Clip DSGD to visible range
    dg_clipped = [min(v, 1e8) for v in dg_vals]
    ax.semilogy([float(s) for s in sigmas], dg_clipped, "s--", color="#e74c3c", label="DSGD (no scheduler)")
    ax.set_xlabel("Noise Scale σ")
    ax.set_ylabel("Mean Final Loss (log scale)")
    ax.set_title("Claim 4: DSignSGD Converges Under Student-t ν=1 Noise (1D Quartic)")
    ax.legend()
    ax.set_ylim(bottom=1e-4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "claim4_dsignsgd.png")
    plt.close(fig)
    print("  Saved claim4_dsignsgd.png")


def plot_claim5():
    """Figure 5: MLP training — DSGD fixed vs adaptive."""
    r = load("claim5_mlp_results.json")
    sigmas = sorted(r["dsgd"].keys(), key=float)
    fixed = [r["dsgd"][s]["mean_final_loss"] for s in sigmas]
    adaptive = [r["dsgd_adaptive"][s]["mean_final_loss"] for s in sigmas]

    fig, ax = plt.subplots()
    # Clip for visibility
    fixed_clipped = [min(v, 1e6) for v in fixed]
    adaptive_clipped = [min(v, 1e6) for v in adaptive]
    ax.semilogy([float(s) for s in sigmas], fixed_clipped, "s--", color="#e74c3c", label="DSGD (fixed stepsize)")
    ax.semilogy([float(s) for s in sigmas], adaptive_clipped, "o-", color="#2ecc71", label="DSGD (adaptive stepsize)")
    ax.set_xlabel("Noise Scale σ")
    ax.set_ylabel("Mean Final Loss (log scale)")
    ax.set_title("Claim 5: MLP Empirical — Adaptive Stepsize Stabilizes Training")
    ax.legend()
    ax.set_ylim(bottom=1e-2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "claim5_mlp.png")
    plt.close(fig)
    print("  Saved claim5_mlp.png")


if __name__ == "__main__":
    print("Generating plots...")
    plot_claim1()
    plot_claim2()
    plot_claim3()
    plot_claim4()
    plot_claim5()
    print("\nAll plots saved to plots/")
