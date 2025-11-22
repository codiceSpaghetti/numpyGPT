from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray


class Module(ABC):
    def __init__(self) -> None:
        self.training: bool = True

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def params(self) -> dict[str, ndarray]:
        return {}

    def grads(self) -> dict[str, ndarray | None]:
        return {}

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
