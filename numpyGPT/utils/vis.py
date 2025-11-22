import json
import os

import matplotlib.pyplot as plt
import numpy as np


class MetricsLogger:
    def __init__(self, log_file: str = 'metrics.json') -> None:
        self.log_file: str = log_file
        self.metrics: dict[str, list[int | float | None]] = {
            'iterations': [],
            'train_loss': [],
            'val_loss': [],
            'grad_norm': [],
            'lr': []
        }

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.metrics = json.load(f)

    def log(self, iter_num: int, train_loss: float | None = None, val_loss: float | None = None, grad_norm: float | None = None, lr: float | None = None) -> None:
        self.metrics['iterations'].append(iter_num)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['lr'].append(lr)

        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f)

    def plot(self, save_path: str = 'training_curves.png') -> str:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        iters = np.array(self.metrics['iterations'])

        axes[0, 0].plot(iters, self.metrics['train_loss'], 'b-', alpha=0.7, label='train')
        if any(v is not None for v in self.metrics['val_loss']):
            val_iters = [i for i, v in zip(iters, self.metrics['val_loss']) if v is not None]
            val_losses = [v for v in self.metrics['val_loss'] if v is not None]
            axes[0, 0].plot(val_iters, val_losses, 'r-', alpha=0.7, label='val')
        axes[0, 0].set_xlabel('iteration')
        axes[0, 0].set_ylabel('loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if any(v is not None for v in self.metrics['grad_norm']):
            grad_iters = [i for i, v in zip(iters, self.metrics['grad_norm']) if v is not None]
            grad_norms = [v for v in self.metrics['grad_norm'] if v is not None]
            axes[0, 1].plot(grad_iters, grad_norms, 'g-', alpha=0.7)
            axes[0, 1].set_xlabel('iteration')
            axes[0, 1].set_ylabel('grad_norm')
            axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(iters, self.metrics['lr'], 'orange', alpha=0.7)
        axes[1, 0].set_xlabel('iteration')
        axes[1, 0].set_ylabel('learning_rate')
        axes[1, 0].grid(True, alpha=0.3)

        if any(v is not None for v in self.metrics['val_loss']):
            val_iters = [i for i, v in zip(iters, self.metrics['val_loss']) if v is not None]
            val_losses = [v for v in self.metrics['val_loss'] if v is not None]
            if len(val_losses) > 1:
                axes[1, 1].plot(val_iters[1:], np.diff(val_losses), 'purple', alpha=0.7)
                axes[1, 1].set_xlabel('iteration')
                axes[1, 1].set_ylabel('val_loss_diff')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path
