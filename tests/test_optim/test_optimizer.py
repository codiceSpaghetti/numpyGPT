import os
import sys
import unittest

import numpy as np

from numpyGPT.nn.modules import Linear
from numpyGPT.optim import Adam, Optimizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.layer = Linear(3, 2)
        self.layer.dW = np.random.randn(3, 2)
        self.layer.db = np.random.randn(2)

    def test_optimizer_interface(self):
        with self.assertRaises(TypeError):
            Optimizer([self.layer])

        class DummyOptimizer(Optimizer):
            def step(self):
                pass

        optimizer = DummyOptimizer([self.layer], lr=0.01)
        self.assertEqual(optimizer.lr, 0.01)
        self.assertEqual(len(optimizer.params), 1)

    def test_zero_grad(self):
        optimizer = Adam([self.layer])

        self.assertIsNotNone(self.layer.dW)
        self.assertIsNotNone(self.layer.db)

        optimizer.zero_grad()

        self.assertTrue(np.allclose(self.layer.dW, 0))
        self.assertTrue(np.allclose(self.layer.db, 0))


class TestAdam(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.layer = Linear(3, 2)
        self.layer.dW = np.random.randn(3, 2) * 0.1
        self.layer.db = np.random.randn(2) * 0.1

    def test_adam_initialization(self):
        optimizer = Adam([self.layer], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        self.assertEqual(optimizer.lr, 0.01)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
        self.assertEqual(optimizer.eps, 1e-8)
        self.assertEqual(optimizer.t, 0)

        self.assertEqual(len(optimizer.m), 1)
        self.assertEqual(len(optimizer.v), 1)

        self.assertTrue(np.allclose(optimizer.m[0]['W'], 0))
        self.assertTrue(np.allclose(optimizer.m[0]['b'], 0))
        self.assertTrue(np.allclose(optimizer.v[0]['W'], 0))
        self.assertTrue(np.allclose(optimizer.v[0]['b'], 0))

    def test_adam_step(self):
        optimizer = Adam([self.layer], lr=0.1)

        W_before = self.layer.W.copy()
        b_before = self.layer.b.copy()

        optimizer.step()

        self.assertEqual(optimizer.t, 1)
        self.assertFalse(np.allclose(self.layer.W, W_before))
        self.assertFalse(np.allclose(self.layer.b, b_before))

        self.assertFalse(np.allclose(optimizer.m[0]['W'], 0))
        self.assertFalse(np.allclose(optimizer.m[0]['b'], 0))
        self.assertFalse(np.allclose(optimizer.v[0]['W'], 0))
        self.assertFalse(np.allclose(optimizer.v[0]['b'], 0))

    def test_adam_multiple_steps(self):
        optimizer = Adam([self.layer], lr=0.01)

        W_history = [self.layer.W.copy()]

        for i in range(5):
            self.layer.dW = np.random.randn(3, 2) * 0.01
            self.layer.db = np.random.randn(2) * 0.01
            optimizer.step()
            W_history.append(self.layer.W.copy())

        self.assertEqual(optimizer.t, 5)

        for i in range(len(W_history) - 1):
            self.assertFalse(np.allclose(W_history[i], W_history[i + 1]))

    def test_adam_convergence(self):
        layer = Linear(2, 1)
        layer.W = np.array([[0.5], [0.5]])
        layer.b = np.array([0.5])

        optimizer = Adam([layer], lr=0.1)

        X = np.array([[1.0, 1.0], [2.0, 2.0]])
        y_true = np.array([[2.0], [4.0]])

        losses = []
        for _ in range(10):
            y_pred = layer.forward(X)
            loss = np.mean((y_pred - y_true) ** 2)
            losses.append(loss)

            dY = 2 * (y_pred - y_true) / len(X)
            layer.backward(dY)
            optimizer.step()

        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
