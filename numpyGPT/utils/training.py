import logging
import sys
import time

import numpy as np

from numpyGPT.optim import Optimizer

from ..nn.modules.module import Module


def setup_logger(name: str = "train", level: int = logging.INFO) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def clip_grad_norm(model: Module, max_norm: float) -> float:
    total_norm = 0.0
    grads = model.grads()

    for _, grad in grads.items():
        if grad is not None:
            param_norm = float(np.linalg.norm(grad))
            total_norm += param_norm**2

    total_norm = float(np.sqrt(total_norm))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for _, grad in grads.items():
            if grad is not None:
                grad *= clip_coef

    return float(total_norm)


def get_lr(optimizer: Optimizer) -> float:
    return optimizer.lr


class TrainingMonitor:
    def __init__(self, log_interval: int = 100) -> None:
        self.log_interval: int = log_interval
        self.step_times: list[float] = []
        self.losses: list[float] = []

    def log_step(
        self, iter_num: int, loss: float, lr: float, grad_norm: float | None = None
    ) -> str | None:
        if iter_num % self.log_interval == 0:
            step_time = (
                time.time() if len(self.step_times) == 0 else time.time() - self.step_times[-1]
            )
            self.step_times.append(time.time())
            self.losses.append(loss)

            msg = f"iter {iter_num:6d} | loss {loss:.4f} | lr {lr:.2e}"
            if grad_norm is not None:
                msg += f" | grad_norm {grad_norm:.2f}"
            if len(self.step_times) > 1:
                msg += f" | ms/step {step_time*1000/self.log_interval:.1f}"

            return msg
        return None

    def get_avg_loss(self, window: int = 100) -> float:
        if len(self.losses) == 0:
            return 0.0
        return float(np.mean(self.losses[-window:]))
