from abc import ABC, abstractmethod


class LRScheduler(ABC):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    @abstractmethod
    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()[0]
        self.optimizer.lr = lr
