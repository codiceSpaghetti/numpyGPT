import numpy as np

from .module import Module


class PositionalEncoding(Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.W = np.random.randn(max_len, d_model) * 0.02
        self.dW = None
        self.cache_input = None

    def forward(self, X):
        B, T, C = X.shape
        self.cache_input = np.arange(T)
        pos_emb = self.W[:T]
        return X + pos_emb

    def backward(self, dZ):
        T = len(self.cache_input)
        self.dW = np.zeros_like(self.W)
        self.dW[:T] = np.sum(dZ, axis=0)
        return dZ

    def params(self):
        return {"W": self.W}

    def grads(self):
        return {"W": self.dW}
