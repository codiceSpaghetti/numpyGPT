import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # He/Kaiming initialization (https://arxiv.org/abs/1502.01852) for ReLU activations: scale = sqrt(2/fan_in)
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        self.dW = None
        self.db = None
        self.cache_input = None

    def forward(self, X):
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

    def backward(self, dZ):
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

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}
