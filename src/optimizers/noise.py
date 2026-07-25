"""Noise models for gradient perturbation."""

import numpy as np


class AffineGaussianNoise:
    """Gaussian noise with variance sigma0^2 + sigma1^2 * ||grad||^2.

    This models the (sigma0^2, sigma1^2)-variance assumption from the paper.
    """

    def __init__(self, sigma0_sq: float = 0.0, sigma1_sq: float = 0.01):
        self.sigma0_sq = sigma0_sq
        self.sigma1_sq = sigma1_sq

    def __call__(self, grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        grad_norm_sq = np.sum(grad**2)
        noise_var = self.sigma0_sq + self.sigma1_sq * grad_norm_sq
        return rng.normal(0, np.sqrt(max(noise_var, 1e-30)), size=grad.shape)


class StudentTNoise:
    """Student-t noise with degrees of freedom nu.

    When nu <= 1, the noise has unbounded expectation.
    """

    def __init__(self, nu: float = 1.0, scale: float = 1.0):
        self.nu = nu
        self.scale = scale

    def __call__(self, grad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_t(self.nu, size=grad.shape)
        return self.scale * z
