import unittest

import numpy as np

from numpyGPT.nn.functional import cross_entropy_loss


class TestCrossEntropyLoss(unittest.TestCase):

    def test_cross_entropy_basic(self):
        probs = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.6, 0.2, 0.1]
        ])
        logits = np.log(probs + 1e-8)
        Y_true = np.array([0, 2, 1])

        loss = cross_entropy_loss(logits, Y_true)
        expected = -(np.log(0.7) + np.log(0.7) + np.log(0.6)) / 3

        self.assertAlmostEqual(loss, expected, places=6)
        self.assertGreater(loss, 0)

    def test_cross_entropy_perfect(self):
        logits = np.array([
            [10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0]
        ])
        Y_true = np.array([0, 2])

        loss = cross_entropy_loss(logits, Y_true)
        self.assertAlmostEqual(loss, 0.0, places=2)

    def test_cross_entropy_numerical_stability(self):
        logits = np.array([
            [-15.0, 1.0, -10.0],
            [-15.0, -10.0, 1.0]
        ])
        Y_true = np.array([0, 0])

        loss = cross_entropy_loss(logits, Y_true)

        self.assertGreater(loss, 15.0)
        self.assertGreater(loss, 0)


if __name__ == '__main__':
    unittest.main()


