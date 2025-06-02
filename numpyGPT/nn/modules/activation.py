import numpy as np

from .module import Module


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.cache_output = None

    def forward(self, X):
        X_shifted = X - np.max(X, axis=-1, keepdims=True)  # (*, d)
        exp_X = np.exp(X_shifted)  # (*, d)
        softmax_output = exp_X / np.sum(exp_X, axis=-1, keepdims=True)  # (*, d)
        self.cache_output = softmax_output
        return softmax_output

    def backward(self, dZ_or_Y_true, Y_true=None):
        if Y_true is not None:
            Y_hat = self.cache_output
            Y = np.zeros_like(Y_hat)
            Y[np.arange(Y_true.size), Y_true] = 1
            # CrossEntropy + Softmax derivative simplifies beautifully to:
            # Note: PyTorch's cross_entropy averages over batch, so we need to divide by batch size
            batch_size = Y_hat.shape[0]
            return (Y_hat - Y) / batch_size  # ∂(CE○Softmax)/∂x = (ŷ - y) / N  (see: https://www.parasdahal.com/softmax-crossentropy)
        else:
            # Otherwise, compute gradient using the full Softmax Jacobian to backpropagate through softmax outputs
            # (see: https://tombolton.io/2018/08/25/softmax-back-propagation-solved-i-think/)
            dZ = dZ_or_Y_true
            softmax_output = self.cache_output
            # Softmax Jacobian: ∂σ_i/∂x_j = σ_i(δ_ij - σ_j)
            return softmax_output * (dZ - np.sum(dZ * softmax_output, axis=-1, keepdims=True))  # ∂L/∂x_i = σ_i(∂L/∂σ_i - Σ_j ∂L/∂σ_j·σ_j)

    def params(self):
        return {}

    def grads(self):
        return {}


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return np.maximum(0, X)  # ReLU(x) = max(0, x)

    def backward(self, dZ):
        X = self.cache_input
        # ReLU derivative: 1 if x > 0, else 0
        grad = (X > 0).astype(np.float32)  # ∂ReLU/∂x = 1 if x > 0, else 0
        return dZ * grad  # ∂L/∂x = ∂L/∂ReLU · ∂ReLU/∂x

    def params(self):
        return {}

    def grads(self):
        return {}


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return np.where(X > 0, X, self.alpha * X)  # LeakyReLU(x) = max(αx, x)

    def backward(self, dZ):
        X = self.cache_input
        # LeakyReLU derivative: 1 if x > 0, else α
        grad = np.where(X > 0, 1.0, self.alpha)  # ∂LeakyReLU/∂x = 1 if x > 0, else α
        return dZ * grad  # ∂L/∂x = ∂L/∂LeakyReLU · ∂LeakyReLU/∂x

    def params(self):
        return {}

    def grads(self):
        return {}
