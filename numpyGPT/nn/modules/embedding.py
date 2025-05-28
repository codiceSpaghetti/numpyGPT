import numpy as np

from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.W = np.random.randn(num_embeddings, embedding_dim) * 0.02
        self.dW = None
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return self.W[X]

    def backward(self, dZ):
        X = self.cache_input
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, X, dZ)
        return None

    def params(self):
        return {"W": self.W}

    def grads(self):
        return {"W": self.dW}
