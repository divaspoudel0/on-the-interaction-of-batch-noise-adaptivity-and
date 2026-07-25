"""Distributed Compressed SGD (DCSGD) optimizer.

Implements gradient descent with:
- Random sparsification compression
- Affine-variance Gaussian noise
- Optional normalization scheduler (Eq. 15 from paper)
"""

import numpy as np
from typing import Optional, Callable


class DCSGD:
    def __init__(
        self,
        lr: float = 0.01,
        omega: float = 4.0,
        noise: Optional[Callable] = None,
        normalize: bool = False,
    ):
        self.lr = lr
        self.omega = omega  # Compression rate: keep 1/omega fraction
        self.noise = noise
        self.normalize = normalize

    def compress(self, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random sparsification: keep each coord with prob 1/omega, scale by omega."""
        d = len(v)
        keep_prob = 1.0 / self.omega
        mask = (rng.random(d) < keep_prob).astype(float)
        return v * mask * self.omega

    def step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        grad_norm = np.linalg.norm(grad)

        # Add noise
        if self.noise is not None:
            g = grad + self.noise(grad, rng)
        else:
            g = grad.copy()

        # Apply compression
        g = self.compress(g, rng)

        # Apply normalization if enabled
        if self.normalize and grad_norm > 0:
            g = g / grad_norm

        return x - self.lr * g

    def run(
        self,
        objective,
        x0: np.ndarray,
        n_steps: int,
        n_runs: int,
        seed: int = 42,
    ) -> dict:
        rng = np.random.default_rng(seed)
        all_losses = np.zeros((n_runs, n_steps))

        for run in range(n_runs):
            x = x0.copy() + rng.standard_normal(x0.shape) * 0.5
            for t in range(n_steps):
                grad = objective.gradient(x)
                x = self.step(x, grad, t, rng)
                all_losses[run, t] = objective.value(x)
                if all_losses[run, t] > 1e15:
                    all_losses[run, t:] = 1e15
                    break

        return {
            "mean_loss": float(np.mean(all_losses[:, -100:], axis=1).mean()),
            "median_loss": float(np.median(all_losses[:, -100:])),
            "divergence_rate": float(np.mean(all_losses[:, -1] > 1e10)),
            "losses": all_losses,
        }
