#! /usr/bin/env python3

import argparse
import os

from numpyGPT.utils.vis import MetricsLogger


def plot_metrics(out_dir: str = 'out/char', output_file: str | None = None) -> None:
    metrics_file = os.path.join(out_dir, 'metrics.json')

    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        exit(1)

    if output_file is None:
        output_file = os.path.join(out_dir, 'training_curves.png')

    metrics = MetricsLogger(metrics_file)
    plot_path = metrics.plot(output_file)
    print(f"Training curves saved to {plot_path}")

    num_iters = len(metrics.metrics['iterations'])
    if num_iters > 0:
        latest_iter = metrics.metrics['iterations'][-1]
        latest_loss = metrics.metrics['train_loss'][-1]
        if latest_loss is not None:
            print(f"Latest: iter {latest_iter}, loss {latest_loss:.4f}")

        val_losses = [v for v in metrics.metrics['val_loss'] if v is not None]
        if val_losses:
            print(f"Best val loss: {min(val_losses):.4f}")
    else:
        print("No metrics found")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='out/char', help='output directory containing metrics.json')
    parser.add_argument('--output_file', default=None, help='output plot file path')
    args = parser.parse_args()

    plot_metrics(args.out_dir, args.output_file)
