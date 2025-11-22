from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class StepLR(LRScheduler):
    """
    Step learning rate scheduler: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L432
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        self.step_size: int = step_size
        self.gamma: float = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [self.base_lr * self.gamma ** (self.last_epoch // self.step_size)]
