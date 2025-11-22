import numpy as np
from numpy import ndarray

from .activation import Softmax
from .linear import Linear
from .module import Module


class MultiHeadAttention(Module):
    """
    Multi-Head Attention (Attention is All You Need <3): https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        self.W_q: Linear = Linear(d_model, d_model)
        self.W_k: Linear = Linear(d_model, d_model)
        self.W_v: Linear = Linear(d_model, d_model)
        self.W_o: Linear = Linear(d_model, d_model)
        self.softmax: Softmax = Softmax()

        self.cache: dict[str, ndarray | tuple] = {}

    def forward(self, X: ndarray, mask: ndarray | None = None) -> ndarray:
        B, T, C = X.shape

        Q = self.W_q(X)  # (B, T, C)
        K = self.W_k(X)  # (B, T, C)
        V = self.W_v(X)  # (B, T, C)

        Q = Q.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)  # (B, nh, T, d_k)
        K = K.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)  # (B, nh, T, d_k)
        V = V.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)  # (B, nh, T, d_k)

        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)  # (B, nh, T, T)

        if mask is not None:
            scores = scores + mask  # (B, nh, T, T)

        original_shape = scores.shape
        scores_reshaped = scores.reshape(-1, scores.shape[-1])  # (B*nh*T, T)
        attn_weights_reshaped = self.softmax(scores_reshaped)  # (B*nh*T, T)
        attn_weights = attn_weights_reshaped.reshape(original_shape)  # (B, nh, T, T)

        attn_output = attn_weights @ V  # (B, nh, T, d_k)

        attn_output = attn_output.swapaxes(1, 2).reshape(B, T, C)  # (B, T, C)
        output = self.W_o(attn_output)  # (B, T, C)

        self.cache = {
            'X': X, 'Q': Q, 'K': K, 'V': V,
            'attn_weights': attn_weights,
            'original_shape': original_shape
        }

        return output

    def backward(self, dZ: ndarray) -> ndarray:
        X, Q, K, V = self.cache['X'], self.cache['Q'], self.cache['K'], self.cache['V']
        attn_weights = self.cache['attn_weights']
        original_shape = self.cache['original_shape']
        B, T, C = X.shape

        dattn_output = self.W_o.backward(dZ)  # ∂L/∂attn_output
        dattn_output = dattn_output.reshape(B, T, self.n_heads, self.d_k).swapaxes(1, 2)  # (B, nh, T, d_k)

        # Attention output = attn_weights @ V
        # ∂L/∂attn_weights = ∂L/∂attn_output @ V^T
        # ∂L/∂V = attn_weights^T @ ∂L/∂attn_output
        dattn_weights = dattn_output @ V.swapaxes(-2, -1)  # ∂L/∂attn_weights
        dV = attn_weights.swapaxes(-2, -1) @ dattn_output  # ∂L/∂V

        dattn_weights_reshaped = dattn_weights.reshape(-1, dattn_weights.shape[-1])
        dscores_reshaped = self.softmax.backward(dattn_weights_reshaped)
        dscores = dscores_reshaped.reshape(original_shape)

        # scores = QK^T / √d_k
        # ∂L/∂Q = ∂L/∂scores @ K / √d_k
        # ∂L/∂K = Q^T @ ∂L/∂scores / √d_k
        dQ = dscores @ K / np.sqrt(self.d_k)  # ∂L/∂Q
        dK = Q.swapaxes(-2, -1) @ dscores / np.sqrt(self.d_k)  # ∂L/∂K

        dQ = dQ.swapaxes(1, 2).reshape(B, T, C)
        dK = dK.swapaxes(1, 2).reshape(B, T, C)
        dV = dV.swapaxes(1, 2).reshape(B, T, C)

        # X branches into Q, K, V - sum gradients from all paths
        dX_q = self.W_q.backward(dQ)  # ∂L/∂X via Q
        dX_k = self.W_k.backward(dK)  # ∂L/∂X via K
        dX_v = self.W_v.backward(dV)  # ∂L/∂X via V

        return dX_q + dX_k + dX_v  # ∂L/∂X = sum of all paths

    def params(self) -> dict[str, ndarray]:
        params = {}
        params.update({f"W_q.{k}": v for k, v in self.W_q.params().items()})
        params.update({f"W_k.{k}": v for k, v in self.W_k.params().items()})
        params.update({f"W_v.{k}": v for k, v in self.W_v.params().items()})
        params.update({f"W_o.{k}": v for k, v in self.W_o.params().items()})
        return params

    def grads(self) -> dict[str, ndarray | None]:
        grads = {}
        grads.update({f"W_q.{k}": v for k, v in self.W_q.grads().items()})
        grads.update({f"W_k.{k}": v for k, v in self.W_k.grads().items()})
        grads.update({f"W_v.{k}": v for k, v in self.W_v.grads().items()})
        grads.update({f"W_o.{k}": v for k, v in self.W_o.grads().items()})
        return grads
