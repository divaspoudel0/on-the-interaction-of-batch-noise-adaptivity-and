#!/usr/bin/env python3
"""
Run all experiments and save results.

This script runs all 5 claim experiments and generates a summary.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiments.claim1_stepsize import run_experiment as run_claim1
from experiments.claim2_compression import run_experiment as run_claim2
from experiments.claim3_normalization import run_experiment as run_claim3
from experiments.claim4_dsignsgd import run_experiment as run_claim4
from experiments.claim5_mlp import run_experiment as run_claim5
from utils import save_results


def main():
    print("=" * 70)
    print("REPRODUCTION: On the Interaction of Batch Noise, Adaptivity,")
    print("and Compression, under (L0,L1)-Smoothness: An SDE Approach")
    print("arXiv 2506.00181 - ICML 2026")
    print("=" * 70)

    start_time = time.time()
    summary = {"claims": {}, "all_supported": True}

    # Run each experiment
    for name, run_fn in [
        ("Claim 1: DSGD Adaptive Stepsize", run_claim1),
        ("Claim 2: DCSGD Compression", run_claim2),
        ("Claim 3: Gradient Normalization", run_claim3),
        ("Claim 4: DSignSGD Heavy-Tailed", run_claim4),
        ("Claim 5: MLP Empirical", run_claim5),
    ]:
        print(f"\n{'='*70}")
        print(f"Running {name}...")
        print(f"{'='*70}")
        try:
            results = run_fn()
            supported = results.get("claim_supported", False)
            summary["claims"][name] = {
                "supported": supported,
                "key_results": {k: v for k, v in results.items() if k != "claim_supported"},
            }
            if not supported:
                summary["all_supported"] = False
        except Exception as e:
            print(f"ERROR: {e}")
            summary["claims"][name] = {"supported": False, "error": str(e)}
            summary["all_supported"] = False

    elapsed = time.time() - start_time
    summary["total_time_seconds"] = elapsed

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for name, result in summary["claims"].items():
        status = "SUPPORTED" if result.get("supported") else "NOT SUPPORTED"
        print(f"  {name}: {status}")
    print(f"\nAll claims supported: {summary['all_supported']}")
    print(f"Total time: {elapsed:.1f} seconds")

    save_results(summary, "experiments/outputs/summary.json")
    print(f"\nResults saved to experiments/outputs/")


if __name__ == "__main__":
    main()
