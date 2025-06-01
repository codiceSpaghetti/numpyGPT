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
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        self.cache_input = X

        if X.ndim > 2:
            X_reshaped = X.reshape(-1, X.shape[-1])
            out = X_reshaped @ self.W + self.b
            return out.reshape(*self.input_shape[:-1], -1)
        else:
            return X @ self.W + self.b

    def backward(self, dZ):
        X = self.cache_input

        if X.ndim > 2:
            X_reshaped = X.reshape(-1, X.shape[-1])
            dZ_reshaped = dZ.reshape(-1, dZ.shape[-1])

            self.dW = X_reshaped.T @ dZ_reshaped
            self.db = np.sum(dZ_reshaped, axis=0)
            dX = dZ_reshaped @ self.W.T
            return dX.reshape(self.input_shape)
        else:
            self.dW = X.T @ dZ
            self.db = np.sum(dZ, axis=0)
            dX = dZ @ self.W.T
            return dX

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}
