"""Distributed SGD (DSGD) optimizer.

Implements gradient descent with optional:
- Gradient noise injection
- Adaptive stepsize (eta / ||grad||)
- Decaying stepsize schedule (eta / sqrt(t+1))
"""

import numpy as np
from typing import Callable, Optional, List


class DSGD:
    def __init__(
        self,
        lr: float = 0.01,
        adaptive: bool = False,
        decay: bool = False,
        noise: Optional[Callable] = None,
    ):
        self.lr = lr
        self.adaptive = adaptive
        self.decay = decay
        self.noise = noise

    def step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        grad_norm = np.linalg.norm(grad)

        # Compute stepsize
        if self.adaptive and grad_norm > 0:
            lr_t = self.lr / grad_norm
        elif self.decay:
            lr_t = self.lr / np.sqrt(t + 1)
        else:
            lr_t = self.lr

        # Add noise
        if self.noise is not None:
            grad = grad + self.noise(grad, rng)

        return x - lr_t * grad

    def run(
        self,
        objective,
        x0: np.ndarray,
        n_steps: int,
        n_runs: int,
        seed: int = 42,
    ) -> dict:
        """Run DSGD and return statistics."""
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
