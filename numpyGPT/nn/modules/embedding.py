import numpy as np
from numpy import ndarray

from .module import Module


class Embedding(Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embed_dim: int = embed_dim

        self.W: ndarray = np.random.randn(vocab_size, embed_dim) * 0.02
        self.dW: ndarray | None = None
        self.cache_input: ndarray | None = None

    def forward(self, X: ndarray) -> ndarray:
        self.cache_input = X  # (B, T)
        out = self.W[X]  # (B, T, embed_dim)
        return out

    def backward(self, dZ: ndarray) -> None:
        self.dW = np.zeros_like(self.W)
        X = self.cache_input

        # We add the gradients to the corresponding rows of the weight matrix W.
        # Since Z[i, j] = W[X[i, j]], the gradient ∂L/∂W[k] is the sum of ∂L/∂Z[i, j]
        # over all (i, j) where X[i, j] == k.
        np.add.at(self.dW, X.flatten(), dZ.reshape(-1, self.embed_dim))  # in place accumulation for efficiency

    def params(self) -> dict[str, ndarray]:
        return {"W": self.W}

    def grads(self) -> dict[str, ndarray | None]:
        return {"W": self.dW}
