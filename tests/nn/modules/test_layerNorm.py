import unittest

import numpy as np

from numpyGPT.nn.modules.layerNorm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_forward(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4

        layer = LayerNorm(hidden_dim)
        X = np.random.randn(batch_size, seq_len, hidden_dim)

        out = layer(X)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_dim))

        X_norm = layer.cache["X_norm"]
        mean = np.mean(X_norm, axis=-1)
        var = np.var(X_norm, axis=-1)
        self.assertTrue(np.allclose(mean, np.zeros_like(mean), atol=1e-6))
        self.assertTrue(np.allclose(var, np.ones_like(var), atol=1e-3))

    def test_layernorm_backward(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4

        layer = LayerNorm(hidden_dim)
        X = np.random.randn(batch_size, seq_len, hidden_dim)
        dZ = np.random.randn(batch_size, seq_len, hidden_dim)

        layer(X)
        dX = layer.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, seq_len, hidden_dim))
        self.assertEqual(layer.dgamma.shape, (hidden_dim,))
        self.assertEqual(layer.dbeta.shape, (hidden_dim,))

        params = layer.params()
        grads = layer.grads()
        self.assertIn("gamma", params)
        self.assertIn("beta", params)
        self.assertIn("gamma", grads)
        self.assertIn("beta", grads)


if __name__ == "__main__":
    unittest.main()
