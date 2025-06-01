#! /usr/bin/env python3

import os

from numpyGPT.utils.vis import MetricsLogger

out_dir = 'out'
metrics_file = os.path.join(out_dir, 'metrics.json')

if not os.path.exists(metrics_file):
    print(f"metrics file not found: {metrics_file}")
    exit(1)

metrics = MetricsLogger(metrics_file)
plot_path = metrics.plot(os.path.join(out_dir, 'training_curves.png'))
print(f"training curves saved to {plot_path}")

num_iters = len(metrics.metrics['iterations'])
if num_iters > 0:
    latest_iter = metrics.metrics['iterations'][-1]
    latest_loss = metrics.metrics['train_loss'][-1]
    if latest_loss is not None:
        print(f"latest: iter {latest_iter}, loss {latest_loss:.4f}")

    val_losses = [v for v in metrics.metrics['val_loss'] if v is not None]
    if val_losses:
        print(f"best val loss: {min(val_losses):.4f}")
else:
    print("no metrics found")
