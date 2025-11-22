import numpy as np
from numpy import ndarray

from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Xavier/Glorot, Best for tanh/sigmoid, https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # He/Kaiming, Best for ReLU/LeakyReLU, https://arxiv.org/abs/1502.01852
        # Lecun, Best for linear activations or SELU, https://arxiv.org/abs/2406.00348

        # Using He/Kaiming since we have ReLU and LeakyReLU activations
        scale = np.sqrt(2.0 / in_features)

        self.W: ndarray = np.random.randn(in_features, out_features) * scale
        self.b: ndarray = np.zeros(out_features)

        self.dW: ndarray | None = None
        self.db: ndarray | None = None
        self.cache_input: ndarray | None = None

    def forward(self, X: ndarray) -> ndarray:
        self.cache_input = X

        if X.ndim == 3:
            B, T, C = X.shape
            X_reshaped = X.reshape(-1, C)  # (B*T, C)
            out = X_reshaped @ self.W + self.b  # (B*T, out_features)
            out = out.reshape(B, T, self.out_features)  # (B, T, out_features)
            return out
        else:
            out = X @ self.W + self.b  # (B, out_features)
            return out

    def backward(self, dZ: ndarray) -> ndarray:
        X = self.cache_input

        if X.ndim == 3:
            B, T, C = X.shape
            X_reshaped = X.reshape(-1, C)
            dZ_reshaped = dZ.reshape(-1, self.out_features)

            # Y = XW + b, so ∂Y/∂W = X^T, ∂Y/∂b = I, ∂Y/∂X = W^T
            self.dW = X_reshaped.T @ dZ_reshaped  # ∂L/∂W = X^T @ ∂L/∂Y
            self.db = np.sum(dZ_reshaped, axis=0)  # ∂L/∂b = Σ ∂L/∂Y

            dX_reshaped = dZ_reshaped @ self.W.T  # ∂L/∂X = ∂L/∂Y @ W^T
            dX = dX_reshaped.reshape(B, T, C)
            return dX
        else:
            self.dW = X.T @ dZ  # ∂L/∂W = X^T @ ∂L/∂Y
            self.db = np.sum(dZ, axis=0)  # ∂L/∂b = Σ ∂L/∂Y
            return dZ @ self.W.T  # ∂L/∂X = ∂L/∂Y @ W^T

    def params(self) -> dict[str, ndarray]:
        return {"W": self.W, "b": self.b}

    def grads(self) -> dict[str, ndarray | None]:
        return {"W": self.dW, "b": self.db}
