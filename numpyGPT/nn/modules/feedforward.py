from .activation import ReLU
from .linear import Linear
from .module import Module


class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, X):
        X = self.linear1(X)  # (B, T, d_ff)
        X = self.relu(X)  # (B, T, d_ff)
        X = self.linear2(X)  # (B, T, d_model)
        return X

    def backward(self, dZ):
        dZ = self.linear2.backward(dZ)
        dZ = self.relu.backward(dZ)
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
