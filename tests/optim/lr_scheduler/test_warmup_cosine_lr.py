import math
import unittest

import numpy as np

from numpyGPT.nn.modules import Linear
from numpyGPT.optim import Adam
from numpyGPT.optim.lr_scheduler import WarmupCosineLR


class TestWarmupCosineLR(unittest.TestCase):
    def setUp(self):
        self.layer = Linear(2, 1)
        self.optimizer = Adam([self.layer], lr=0.1)

    def test_warmup_phase(self):
        scheduler = WarmupCosineLR(self.optimizer, warmup_iters=5, lr_decay_iters=10, min_lr=0.01)

        expected_lrs = []
        for it in range(5):
            expected_lr = 0.1 * (it + 1) / (5 + 1)
            expected_lrs.append(expected_lr)

        for i, expected_lr in enumerate(expected_lrs):
            if i > 0:
                scheduler.step()
            self.assertAlmostEqual(self.optimizer.lr, expected_lr, places=6)

    def test_decay_phase(self):
        scheduler = WarmupCosineLR(self.optimizer, warmup_iters=2, lr_decay_iters=8, min_lr=0.01)

        # Skip warmup phase (go to start of decay)
        for _ in range(3):
            scheduler.step()

        # Test that we're in decay phase and decreasing
        lr_start = self.optimizer.lr
        scheduler.step()
        lr_mid = self.optimizer.lr

        # Should be decreasing
        self.assertLess(lr_mid, lr_start)

        # Should be above min_lr
        self.assertGreater(lr_mid, 0.01)

    def test_min_lr_phase(self):
        scheduler = WarmupCosineLR(self.optimizer, warmup_iters=2, lr_decay_iters=5, min_lr=0.01)

        # Go through warmup and decay phases
        for _ in range(6):
            scheduler.step()

        # Should be at min_lr
        self.assertAlmostEqual(self.optimizer.lr, 0.01, places=6)

        # Should stay at min_lr
        for _ in range(5):
            scheduler.step()
            self.assertAlmostEqual(self.optimizer.lr, 0.01, places=6)

    def test_complete_schedule(self):
        scheduler = WarmupCosineLR(self.optimizer, warmup_iters=3, lr_decay_iters=10, min_lr=0.005)

        lr_history = [self.optimizer.lr]

        for _ in range(15):
            scheduler.step()
            lr_history.append(self.optimizer.lr)

        # Warmup phase: should be increasing
        for i in range(1, 4):
            self.assertGreater(lr_history[i], lr_history[i-1])

        # Should reach approximately base_lr at end of warmup
        self.assertGreater(lr_history[4], 0.09)  # Close to 0.1

        # Decay phase: should be decreasing (stop before reaching min_lr)
        for i in range(4, 10):
            self.assertGreater(lr_history[i], lr_history[i+1])

        # Min lr phase: should stay constant
        for i in range(11, 16):
            self.assertAlmostEqual(lr_history[i], 0.005, places=6)

    def test_edge_cases(self):
        # Test with warmup_iters = 0 (no warmup)
        scheduler = WarmupCosineLR(self.optimizer, warmup_iters=0, lr_decay_iters=5, min_lr=0.01)

        # Should immediately start decay from epoch 0
        scheduler.step()
        decay_ratio = (1 - 0) / (5 - 0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        expected_lr = 0.01 + coeff * (0.1 - 0.01)
        self.assertAlmostEqual(self.optimizer.lr, expected_lr, places=6)

    def test_integration_with_training(self):
        layer = Linear(2, 1)
        optimizer = Adam([layer], lr=0.1)
        scheduler = WarmupCosineLR(optimizer, warmup_iters=2, lr_decay_iters=6, min_lr=0.01)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[3.0], [7.0]])

        lr_history = [optimizer.lr]

        for epoch in range(10):
            y_pred = layer.forward(X)
            loss = np.mean((y_pred - y) ** 2)

            dY = 2 * (y_pred - y) / len(X)
            layer.backward(dY)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            lr_history.append(optimizer.lr)

        # Check that we have warmup, decay, and min lr phases
        warmup_end = lr_history[2]
        decay_start = lr_history[3]
        min_lr_phase = lr_history[7:]

        # Warmup should increase learning rate
        self.assertGreater(warmup_end, lr_history[0])

        # Decay should decrease learning rate
        self.assertLess(lr_history[6], decay_start)

        # Min lr phase should be constant
        for lr in min_lr_phase:
            self.assertAlmostEqual(lr, 0.01, places=6)


if __name__ == '__main__':
    unittest.main()


