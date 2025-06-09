from .lr_scheduler import LRScheduler


class StepLR(LRScheduler):
    """
    Step learning rate scheduler: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/optim/lr_scheduler.py#L432
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.base_lr * self.gamma ** (self.last_epoch // self.step_size)]
