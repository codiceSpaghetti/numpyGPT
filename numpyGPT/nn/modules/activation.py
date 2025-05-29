import numpy as np

from .module import Module


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.cache_output = None

    def forward(self, X):
        X_shifted = X - np.max(X, axis=-1, keepdims=True)
        exp_X = np.exp(X_shifted)
        softmax_output = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        self.cache_output = softmax_output
        return softmax_output

    def backward(self, Y_hat, Y_true):
        Y = np.zeros_like(Y_hat)
        Y[np.arange(Y_true.size), Y_true] = 1
        return Y_hat - Y

    def params(self):
        return {}

    def grads(self):
        return {}
