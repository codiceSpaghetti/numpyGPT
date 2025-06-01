import numpy as np

from .module import Module


class PositionalEncoding(Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.W = np.random.randn(max_len, d_model) * 0.02
        self.dW = None
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        B, T, C = X.shape  # (B, T, C)

        pos_emb = self.W[:T, :]  # (T, C)
        out = X + pos_emb  # (B, T, C)
        return out

    def backward(self, dZ):
        B, T, C = dZ.shape
        self.dW = np.zeros_like(self.W)

        # out = X + W[:T,:], so ∂out/∂W = I, ∂out/∂X = I
        self.dW[:T, :] = np.sum(dZ, axis=0)  # ∂L/∂W = Σ_batch ∂L/∂out we will need it for gradient update
        return dZ  # ∂L/∂X = ∂L/∂out

    def params(self):
        return {"W": self.W}

    def grads(self):
        return {"W": self.dW}
