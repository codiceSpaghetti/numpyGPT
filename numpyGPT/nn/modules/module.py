from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy import ndarray


class Module(ABC):
    def __init__(self) -> None:
        self.training: bool = True

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: ndarray) -> ndarray | None:
        raise NotImplementedError

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __call__(self, *args: Any, **kwargs: Any) -> ndarray:
        return self.forward(*args, **kwargs)
