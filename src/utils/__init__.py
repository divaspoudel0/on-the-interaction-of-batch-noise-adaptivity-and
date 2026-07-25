"""Utility functions for running experiments and saving results."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_results(results: Dict[str, Any], path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    cleaned = json.loads(json.dumps(results, default=convert))
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)
