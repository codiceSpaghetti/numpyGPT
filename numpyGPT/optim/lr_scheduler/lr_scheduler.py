from abc import ABC, abstractmethod

from ...optim.optimizer import Optimizer


class LRScheduler(ABC):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer: Optimizer = optimizer
        self.base_lr: float = optimizer.lr
        self.last_epoch: int = last_epoch

    @abstractmethod
    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()[0]
        self.optimizer.lr = lr
