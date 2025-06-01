import numpy as np

from .activation import GELU, SwiGLU
from .linear import Linear
from .module import Module


class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.gelu = GELU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, X):
        X = self.linear1(X)
        X = self.gelu(X)
        X = self.linear2(X)
        return X

    def backward(self, dZ):
        dZ = self.linear2.backward(dZ)
        dZ = self.gelu.backward(dZ)
        dZ = self.linear1.backward(dZ)
        return dZ

    def params(self):
        params = {}
        params.update({f"linear1.{k}": v for k, v in self.linear1.params().items()})
        params.update({f"linear2.{k}": v for k, v in self.linear2.params().items()})
        return params

    def grads(self):
        grads = {}
        grads.update({f"linear1.{k}": v for k, v in self.linear1.grads().items()})
        grads.update({f"linear2.{k}": v for k, v in self.linear2.grads().items()})
        return grads


class SwiGLUFeedForward(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff * 2)
        self.swiglu = SwiGLU()
        self.down_proj = Linear(d_ff, d_model)

    def forward(self, X):
        X = self.gate_proj(X)
        X = self.swiglu(X)
        X = self.down_proj(X)
        return X

    def backward(self, dZ):
        dZ = self.down_proj.backward(dZ)
        dZ = self.swiglu.backward(dZ)
        dZ = self.gate_proj.backward(dZ)
        return dZ

    def params(self):
        params = {}
        params.update({f"gate_proj.{k}": v for k, v in self.gate_proj.params().items()})
        params.update({f"down_proj.{k}": v for k, v in self.down_proj.params().items()})
        return params

    def grads(self):
        grads = {}
        grads.update({f"gate_proj.{k}": v for k, v in self.gate_proj.grads().items()})
        grads.update({f"down_proj.{k}": v for k, v in self.down_proj.grads().items()})
        return grads
