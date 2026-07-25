"""Quadratic objective: f(x) = 0.5 * x^T A x.

Eigenvalues of A determine the (L0, L1)-smoothness constants.
"""

import numpy as np


class QuadraticObjective:
    def __init__(self, A: np.ndarray):
        self.A = A
        self.d = A.shape[0]
        self.eigenvalues = np.linalg.eigvalsh(A)

    def value(self, x: np.ndarray) -> float:
        return 0.5 * x @ self.A @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    @property
    def L0(self) -> float:
        return float(np.min(self.eigenvalues))

    @property
    def L1(self) -> float:
        # For a linear Hessian, L1 = 0 since eigenvalues are constant
        return 0.0

    @property
    def condition_number(self) -> float:
        return float(np.max(self.eigenvalues) / max(np.min(self.eigenvalues), 1e-10))
