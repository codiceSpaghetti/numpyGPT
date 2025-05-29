import unittest
import numpy as np

from numpyGPT.nn.functional import cross_entropy_loss


class TestCrossEntropyLoss(unittest.TestCase):

    def test_cross_entropy_basic(self):
        Y_hat = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.6, 0.2, 0.1]
        ])
        Y_true = np.array([0, 2, 1])

        loss = cross_entropy_loss(Y_hat, Y_true)
        expected = -(np.log(0.7) + np.log(0.7) + np.log(0.6)) / 3

        self.assertAlmostEqual(loss, expected, places=6)
        self.assertGreater(loss, 0)

    def test_cross_entropy_perfect(self):
        Y_hat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        Y_true = np.array([0, 2])

        loss = cross_entropy_loss(Y_hat, Y_true)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_cross_entropy_numerical_stability(self):
        eps = 1e-7
        Y_hat = np.array([
            [eps, 1.0-eps, 0.0],
            [eps, 0.0, 1.0-eps]
        ])
        Y_true = np.array([0, 0])

        loss = cross_entropy_loss(Y_hat, Y_true)
        expected = -np.log(eps + eps)

        self.assertAlmostEqual(loss, expected, places=6)
        self.assertGreater(loss, 0)


if __name__ == '__main__':
    unittest.main()
