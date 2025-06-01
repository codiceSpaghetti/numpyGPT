import numpy as np

from .module import Module


class LayerNorm(Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        self.dgamma = None
        self.dbeta = None
        self.cache = {}

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)  # (B, T, 1)
        var = np.var(X, axis=-1, keepdims=True)  # (B, T, 1)

        X_norm = (X - mean) / np.sqrt(var + self.eps)  # (B, T, C)

        out = self.gamma * X_norm + self.beta  # (B, T, C)

        self.cache = {
            'X': X,
            'mean': mean,
            'var': var,
            'X_norm': X_norm
        }

        return out

    def backward(self, dZ):
        X = self.cache['X']
        mean = self.cache['mean']
        var = self.cache['var']
        X_norm = self.cache['X_norm']

        N = X.shape[-1]

        self.dgamma = np.sum(dZ * X_norm, axis=(0, 1))  # ∂L/∂γ = Σ ∂L/∂y · x̂
        self.dbeta = np.sum(dZ, axis=(0, 1))  # ∂L/∂β = Σ ∂L/∂y

        dX_norm = dZ * self.gamma  # ∂L/∂x̂ = ∂L/∂y · γ

        # x̂ = (x-μ)/√(σ²+ε), so ∂x̂/∂σ² = (x-μ)·(-1/2)(σ²+ε)^(-3/2)
        dvar = np.sum(dX_norm * (X - mean), axis=-1, keepdims=True) * -1/2 * (var + self.eps)**(-3/2)  # ∂L/∂σ²

        # μ = (1/N)Σx, so ∂μ/∂x = 1/N
        # μ affects x̂ directly: ∂x̂/∂μ = -1/√(σ²+ε)
        # μ affects σ² indirectly: σ² = (1/N)Σ(x-μ)², so ∂σ²/∂μ = (1/N)Σ(-2(x-μ)) = -2(x-μ)
        dmean = np.sum(dX_norm, axis=-1, keepdims=True) * -1 / np.sqrt(var + self.eps) + \
            dvar * np.sum(-2 * (X - mean), axis=-1, keepdims=True) / N  # ∂L/∂μ

        # Total: ∂L/∂x = ∂L/∂x̂·∂x̂/∂x + ∂L/∂μ·∂μ/∂x + ∂L/∂σ²·∂σ²/∂x
        dX = dX_norm / np.sqrt(var + self.eps) + dvar * 2 * (X - mean) / N + dmean / N  # ∂L/∂x

        return dX

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}
