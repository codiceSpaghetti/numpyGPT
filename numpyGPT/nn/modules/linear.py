import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((out_dim,))
        self.dW = None
        self.db = None
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return X @ self.W + self.b

    def backward(self, dZ):
        X = self.cache_input
        self.dW = X.T @ dZ
        self.db = np.sum(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}
