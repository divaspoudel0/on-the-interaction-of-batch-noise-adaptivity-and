"""Quartic objective: f(x) = sum(x_j^4) / 4.

This is (L0, L1)-smooth with L0 = 0, L1 = 3.
The Hessian is diag(3 * x_j^2), so ||H||_op = 3 * max(x_j^2).
"""

import numpy as np


class QuarticObjective:
    def __init__(self, d: int):
        self.d = d

    def value(self, x: np.ndarray) -> float:
        return np.sum(x**4) / 4.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return x**3

    def hessian_eigenvalues(self, x: np.ndarray) -> np.ndarray:
        return 3.0 * x**2

    @property
    def L0(self) -> float:
        return 0.0

    @property
    def L1(self) -> float:
        return 3.0
