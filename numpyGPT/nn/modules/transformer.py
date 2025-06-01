import numpy as np

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layerNorm import LayerNorm
from .module import Module


class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)

        self.ln2 = LayerNorm(d_model)

    def forward(self, X, mask=None):
        ln1_out = self.ln1(X)  # (B, T, C)
        attn_out = self.attn(ln1_out, mask)  # (B, T, C)
        X = X + attn_out  # (B, T, C)

        ln2_out = self.ln2(X)  # (B, T, C)
        ffn_out = self.ffn(ln2_out)  # (B, T, C)
        X = X + ffn_out  # (B, T, C)

        return X

    def backward(self, dZ):
        # Residual: y = x + f(x), so ∂L/∂x = ∂L/∂y + ∂L/∂f(x) · ∂f/∂x

        dffn_out = dZ.copy()
        dX1 = dZ.copy()

        dffn_in = self.ffn.backward(dffn_out)
        dln2_out = dffn_in
        dX2 = self.ln2.backward(dln2_out)

        dX1 += dX2  # Sum both paths

        dattn_out = dX1.copy()
        dX3 = dX1.copy()

        dattn_in = self.attn.backward(dattn_out)
        dln1_out = dattn_in
        dX4 = self.ln1.backward(dln1_out)

        dX3 += dX4  # Sum both paths

        return dX3

    def params(self):
        params = {}
        params.update({f"attn.{k}": v for k, v in self.attn.params().items()})
        params.update({f"ln1.{k}": v for k, v in self.ln1.params().items()})
        params.update({f"ffn.{k}": v for k, v in self.ffn.params().items()})
        params.update({f"ln2.{k}": v for k, v in self.ln2.params().items()})
        return params

    def grads(self):
        grads = {}
        grads.update({f"attn.{k}": v for k, v in self.attn.grads().items()})
        grads.update({f"ln1.{k}": v for k, v in self.ln1.grads().items()})
        grads.update({f"ffn.{k}": v for k, v in self.ffn.grads().items()})
        grads.update({f"ln2.{k}": v for k, v in self.ln2.grads().items()})
        return grads
