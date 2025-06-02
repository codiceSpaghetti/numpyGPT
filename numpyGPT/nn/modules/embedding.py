import numpy as np

from .module import Module


class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.W = np.random.randn(vocab_size, embed_dim) * 0.02
        self.dW = None
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X  # (B, T)
        out = self.W[X]  # (B, T, embed_dim)
        return out

    def backward(self, dZ):
        self.dW = np.zeros_like(self.W)
        X = self.cache_input

        # out[i,j] = W[X[i,j]], so ∂L/∂W[k] = Σ_{i,j: X[i,j]=k} ∂L/∂out[i,j]
        np.add.at(self.dW, X.flatten(), dZ.reshape(-1, self.embed_dim))

    def params(self):
        return {"W": self.W}

    def grads(self):
        return {"W": self.dW}
