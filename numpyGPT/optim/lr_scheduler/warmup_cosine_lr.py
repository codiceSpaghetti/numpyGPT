import math

from .lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        # Apply the initial learning rate
        if last_epoch == -1:
            self.last_epoch = 0
            lr = self.get_lr()[0]
            self.optimizer.lr = lr

    def get_lr(self):
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
