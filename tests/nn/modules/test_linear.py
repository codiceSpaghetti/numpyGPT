import unittest

import numpy as np

from numpyGPT.nn.modules.linear import Linear


class TestLinear(unittest.TestCase):
    def test_linear_forward(self):
        batch_size = 2
        in_dim = 3
        out_dim = 4

        layer = Linear(in_dim, out_dim)
        X = np.random.randn(batch_size, in_dim)

        out = layer(X)
        self.assertEqual(out.shape, (batch_size, out_dim))
        self.assertTrue(np.allclose(out, X @ layer.W + layer.b))

    def test_linear_backward(self):
        batch_size = 2
        in_dim = 3
        out_dim = 4

        layer = Linear(in_dim, out_dim)
        X = np.random.randn(batch_size, in_dim)
        dZ = np.random.randn(batch_size, out_dim)

        layer(X)
        dX = layer.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, in_dim))
        self.assertEqual(layer.dW.shape, (in_dim, out_dim))
        self.assertEqual(layer.db.shape, (out_dim,))
        self.assertTrue(np.allclose(dX, dZ @ layer.W.T))

        params = layer.params()
        grads = layer.grads()
        self.assertIn("W", params)
        self.assertIn("b", params)
        self.assertIn("W", grads)
        self.assertIn("b", grads)


if __name__ == '__main__':
    unittest.main()


