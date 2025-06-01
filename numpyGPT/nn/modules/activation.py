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

    def backward(self, dZ_or_Y_true, Y_true=None):
        if Y_true is not None:
            Y_hat = self.cache_output
            Y = np.zeros_like(Y_hat)
            Y[np.arange(Y_true.size), Y_true] = 1
            return Y_hat - Y
        else:
            dZ = dZ_or_Y_true
            softmax_output = self.cache_output
            return softmax_output * (dZ - np.sum(dZ * softmax_output, axis=-1, keepdims=True))

    def params(self):
        return {}

    def grads(self):
        return {}


class GELU(Module):
    def __init__(self):
        super().__init__()
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)))

    def backward(self, dZ):
        X = self.cache_input
        tanh_arg = np.sqrt(2 / np.pi) * (X + 0.044715 * X**3)
        tanh_val = np.tanh(tanh_arg)
        sech2_val = 1 - tanh_val**2

        grad = 0.5 * (1 + tanh_val) + 0.5 * X * sech2_val * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * X**2)
        return dZ * grad

    def params(self):
        return {}

    def grads(self):
        return {}


class SwiGLU(Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, X):
        gate, value = np.split(X, 2, axis=-1)
        swish_gate = gate * self._sigmoid(gate)
        output = swish_gate * value

        self.cache = {
            'gate': gate,
            'value': value,
            'swish_gate': swish_gate,
            'sigmoid_gate': self._sigmoid(gate)
        }

        return output

    def backward(self, dZ):
        gate = self.cache['gate']
        value = self.cache['value']
        swish_gate = self.cache['swish_gate']
        sigmoid_gate = self.cache['sigmoid_gate']

        dvalue = dZ * swish_gate
        dswish_gate = dZ * value

        dgate = dswish_gate * (sigmoid_gate + gate * sigmoid_gate * (1 - sigmoid_gate))

        return np.concatenate([dgate, dvalue], axis=-1)

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))

    def params(self):
        return {}

    def grads(self):
        return {}
