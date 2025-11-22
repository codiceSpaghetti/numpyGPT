import numpy as np
from numpy import ndarray

from .module import Module


class LayerNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps

        self.gamma: ndarray = np.ones(d_model)
        self.beta: ndarray = np.zeros(d_model)

        self.dgamma: ndarray | None = None
        self.dbeta: ndarray | None = None
        self.cache: dict[str, ndarray] = {}

    def forward(self, X: ndarray) -> ndarray:
        mean = np.mean(X, axis=-1, keepdims=True)  # (B, T, 1)
        var = np.var(X, axis=-1, keepdims=True)  # (B, T, 1)

        X_norm = (X - mean) / np.sqrt(var + self.eps)  # z-score normalization

        out = self.gamma * X_norm + self.beta  # (B, T, C)

        self.cache = {
            'X': X,
            'mean': mean,
            'var': var,
            'X_norm': X_norm
        }

        return out

    def backward(self, dZ: ndarray) -> ndarray:
        X = self.cache['X']
        mean = self.cache['mean']
        var = self.cache['var']
        X_norm = self.cache['X_norm']

        N = X.shape[-1]

        # sum over batch and time dimensions
        self.dgamma = np.sum(dZ * X_norm, axis=(0, 1))  # mul -> ∂L/∂γ = Σ ∂L/∂y · x̂
        self.dbeta = np.sum(dZ, axis=(0, 1))  # sum -> ∂L/∂β = Σ ∂L/∂y

        dX_norm = dZ * self.gamma  # mul -> ∂L/∂x̂ = ∂L/∂y · γ

        # x̂ = (x-μ)/√(σ²+ε), so ∂x̂/∂σ² = (x-μ)·(-1/2)(σ²+ε)^(-3/2)
        dvar = np.sum(dX_norm * (X - mean), axis=-1, keepdims=True) * -1/2 * (var + self.eps)**(-3/2)

        # μ = (1/N)Σx, so ∂μ/∂x = 1/N
        # μ affects x̂ directly: x̂ = (x-μ)/√(σ²+ε), so ∂x̂/∂μ = -1/√(σ²+ε)
        # μ affects σ² indirectly: σ² = (1/N)Σ(x-μ)², so ∂σ²/∂μ = (1/N)Σ(-2(x-μ)) = -2(x-μ)
        dmean = np.sum(dX_norm, axis=-1, keepdims=True) * -1 / np.sqrt(var + self.eps) + \
            dvar * np.sum(-2 * (X - mean), axis=-1, keepdims=True) / N

        # x̂ = (x - μ) / √(σ² + ε), so ∂x̂/∂x = 1 / √(σ² + ε)
        # μ = (1/N) Σx, so ∂μ/∂x = 1/N
        # σ² = (1/N) Σ(x - μ)², so ∂σ²/∂x = 2(x - μ)/N
        # Total: ∂L/∂x = ∂L/∂x̂ · ∂x̂/∂x + ∂L/∂μ · ∂μ/∂x + ∂L/∂σ² · ∂σ²/∂x
        dX = dX_norm / np.sqrt(var + self.eps) + dmean / N + dvar * 2 * (X - mean) / N

        return dX

    def params(self) -> dict[str, ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self) -> dict[str, ndarray | None]:
        return {"gamma": self.dgamma, "beta": self.dbeta}
