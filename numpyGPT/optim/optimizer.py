from abc import ABC, abstractmethod

from ..nn.modules.module import Module


class Optimizer(ABC):
    def __init__(self, modules: list[Module], lr: float = 0.001) -> None:
        if self.__class__ == Optimizer:
            raise TypeError("Cannot instantiate abstract class Optimizer directly")
        self.params: list[Module] = modules
        self.lr: float = lr

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for module in self.params:
            grads = module.grads()
            for grad_key in grads:
                grad = grads[grad_key]
                if grad is not None:
                    grad.fill(0.0)
