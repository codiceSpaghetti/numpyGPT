import logging
import sys
import time

import numpy as np


def setup_logger(name='train', level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def clip_grad_norm(model, max_norm):
    total_norm = 0.0
    grads = model.grads()

    for _, grad in grads.items():
        if grad is not None:
            param_norm = np.linalg.norm(grad)
            total_norm += param_norm ** 2

    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for _, grad in grads.items():
            if grad is not None:
                grad *= clip_coef

    return total_norm


def get_lr(optimizer):
    return optimizer.lr


class TrainingMonitor:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_times = []
        self.losses = []

    def log_step(self, iter_num, loss, lr, grad_norm=None):
        if iter_num % self.log_interval == 0:
            step_time = time.time() if len(self.step_times) == 0 else time.time() - self.step_times[-1]
            self.step_times.append(time.time())
            self.losses.append(loss)

            msg = f"iter {iter_num:6d} | loss {loss:.4f} | lr {lr:.2e}"
            if grad_norm is not None:
                msg += f" | grad_norm {grad_norm:.2f}"
            if len(self.step_times) > 1:
                msg += f" | ms/step {step_time*1000/self.log_interval:.1f}"

            return msg
        return None

    def get_avg_loss(self, window=100):
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses[-window:])
