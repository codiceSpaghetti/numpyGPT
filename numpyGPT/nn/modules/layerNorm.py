import numpy as np

from .module import Module


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.dgamma = None
        self.dbeta = None
        self.cache = {}

    def forward(self, X):
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - mu) / np.sqrt(var + self.eps)
        out = self.gamma * X_norm + self.beta

        self.cache = {
            'X': X,
            'X_norm': X_norm,
            'mu': mu,
            'var': var
        }
        return out

    def backward(self, dZ):
        X, X_norm, mu, var = self.cache['X'], self.cache['X_norm'], self.cache['mu'], self.cache['var']
        N = X.shape[-1]

        self.dgamma = np.sum(dZ * X_norm, axis=tuple(range(len(X.shape)-1)))
        self.dbeta = np.sum(dZ, axis=tuple(range(len(X.shape)-1)))

        dX_norm = dZ * self.gamma
        dvar = np.sum(dX_norm * (X - mu) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmu = np.sum(dX_norm * -1/np.sqrt(var + self.eps), axis=-1, keepdims=True) + dvar * np.sum(-2 * (X - mu), axis=-1, keepdims=True) / N

        dX = dX_norm / np.sqrt(var + self.eps) + dvar * 2 * (X - mu) / N + dmu / N
        return dX

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}
