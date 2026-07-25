"""Simple MLP for Claim 5 experiments.

Architecture: h_theta: R^20 -> R, single hidden layer of width 64, ReLU.
Loss: MSE with linear teacher.
"""

import numpy as np
from typing import Tuple


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class SimpleMLP:
    """2-layer MLP: input_dim -> hidden_dim -> 1, ReLU activation."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim,)) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.h = relu(X @ self.W1 + self.b1)
        return (self.h @ self.W2 + self.b2).flatten()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(X)
        return float(np.mean((pred - y) ** 2))

    def compute_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        n = X.shape[0]
        pred = self.forward(X)
        loss = np.mean((pred - y) ** 2)

        dpred = 2 * (pred - y) / n
        dW2 = self.h.T @ dpred.reshape(-1, 1)
        db2 = dpred.sum()
        dh = np.outer(dpred, self.W2)
        dh[self.h <= 0] = 0
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)

        grads = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), [db2]])
        return grads, loss

    def update(self, g: np.ndarray, lr: float):
        idx = 0
        d_in, d_h = self.W1.shape

        self.W1 -= lr * g[idx:idx + d_in * d_h].reshape(d_in, d_h)
        idx += d_in * d_h
        self.b1 -= lr * g[idx:idx + d_h]
        idx += d_h
        self.W2 -= lr * g[idx:idx + d_h]
        idx += d_h
        self.b2 -= lr * g[idx:idx + 1]


def generate_regression_data(
    n_samples: int = 256,
    input_dim: int = 20,
    noise_std: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate regression data with a linear teacher."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, input_dim))
    w_star = rng.standard_normal(input_dim)
    y = X @ w_star + noise_std * rng.standard_normal(n_samples)
    return X, y, w_star


def random_sparsify(v: np.ndarray, omega: float, rng: np.random.Generator) -> np.ndarray:
    """Random sparsification: keep each coord with prob 1/omega, scale by omega."""
    d = len(v)
    keep_prob = 1.0 / omega
    mask = (rng.random(d) < keep_prob).astype(float)
    return v * mask * omega
