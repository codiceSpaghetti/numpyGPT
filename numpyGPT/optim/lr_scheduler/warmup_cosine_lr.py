import math

from ...optim.optimizer import Optimizer
from .lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """
    Cosine Learning Rate Scheduler: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_iters: int = warmup_iters
        self.lr_decay_iters: int = lr_decay_iters
        self.min_lr: float = min_lr
        super().__init__(optimizer, last_epoch)
        # Apply the initial learning rate
        if last_epoch == -1:
            self.last_epoch = 0
            lr = self.get_lr()[0]
            self.optimizer.lr = lr

    def get_lr(self) -> list[float]:
        it = self.last_epoch
        learning_rate = self.base_lr

        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return [learning_rate * (it + 1) / (self.warmup_iters + 1)]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return [self.min_lr]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return [self.min_lr + coeff * (learning_rate - self.min_lr)]
