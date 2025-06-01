import numpy as np

from .activation import Softmax
from .linear import Linear
from .module import Module


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.softmax = Softmax()

        self.cache = {}

    def forward(self, X, mask=None):
        B, T, C = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)
        K = K.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)
        V = V.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)

        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        original_shape = scores.shape
        scores_reshaped = scores.reshape(-1, scores.shape[-1])
        attn_weights_reshaped = self.softmax(scores_reshaped)
        attn_weights = attn_weights_reshaped.reshape(original_shape)

        attn_output = attn_weights @ V

        attn_output = attn_output.swapaxes(1, 2).reshape(B, T, C)
        output = self.W_o(attn_output)

        self.cache = {
            'X': X, 'Q': Q, 'K': K, 'V': V,
            'scores': scores, 'attn_weights': attn_weights,
            'attn_output': attn_output, 'mask': mask,
            'original_shape': original_shape
        }

        return output

    def backward(self, dZ):
        X, Q, K, V = self.cache['X'], self.cache['Q'], self.cache['K'], self.cache['V']
        scores, attn_weights, attn_output = self.cache['scores'], self.cache['attn_weights'], self.cache['attn_output']
        original_shape = self.cache['original_shape']
        B, T, C = X.shape

        dattn_output = self.W_o.backward(dZ)
        dattn_output = dattn_output.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)

        dattn_weights = dattn_output @ V.swapaxes(-2, -1)
        dV = attn_weights.swapaxes(-2, -1) @ dattn_output

        dattn_weights_reshaped = dattn_weights.reshape(-1, dattn_weights.shape[-1])
        dscores_reshaped = self.softmax.backward(dattn_weights_reshaped)
        dscores = dscores_reshaped.reshape(original_shape)

        dQ = dscores @ K / np.sqrt(self.d_k)
        dK = Q.swapaxes(-2, -1) @ dscores / np.sqrt(self.d_k)

        dQ = dQ.swapaxes(1, 2).reshape(B, T, C)
        dK = dK.swapaxes(1, 2).reshape(B, T, C)
        dV = dV.swapaxes(1, 2).reshape(B, T, C)

        dX_q = self.W_q.backward(dQ)
        dX_k = self.W_k.backward(dK)
        dX_v = self.W_v.backward(dV)

        return dX_q + dX_k + dX_v

    def params(self):
        params = {}
        params.update({f"W_q.{k}": v for k, v in self.W_q.params().items()})
        params.update({f"W_k.{k}": v for k, v in self.W_k.params().items()})
        params.update({f"W_v.{k}": v for k, v in self.W_v.params().items()})
        params.update({f"W_o.{k}": v for k, v in self.W_o.params().items()})
        return params

    def grads(self):
        grads = {}
        grads.update({f"W_q.{k}": v for k, v in self.W_q.grads().items()})
        grads.update({f"W_k.{k}": v for k, v in self.W_k.grads().items()})
        grads.update({f"W_v.{k}": v for k, v in self.W_v.grads().items()})
        grads.update({f"W_o.{k}": v for k, v in self.W_o.grads().items()})
        return grads
