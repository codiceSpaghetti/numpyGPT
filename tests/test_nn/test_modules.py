import unittest

import numpy as np

from numpyGPT.nn.modules.activation import LeakyReLU, ReLU, Softmax
from numpyGPT.nn.modules.embedding import Embedding
from numpyGPT.nn.modules.layerNorm import LayerNorm
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


class TestEmbedding(unittest.TestCase):
    def test_embedding_forward(self):
        batch_size = 2
        seq_len = 3
        num_embeddings = 10
        embedding_dim = 4

        layer = Embedding(num_embeddings, embedding_dim)
        X = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))

        out = layer(X)
        self.assertEqual(out.shape, (batch_size, seq_len, embedding_dim))
        self.assertTrue(np.allclose(out, layer.W[X]))

    def test_embedding_backward(self):
        batch_size = 2
        seq_len = 3
        num_embeddings = 10
        embedding_dim = 4

        layer = Embedding(num_embeddings, embedding_dim)
        X = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))
        dZ = np.random.randn(batch_size, seq_len, embedding_dim)

        layer(X)
        dX = layer.backward(dZ)

        self.assertIsNone(dX)
        self.assertEqual(layer.dW.shape, (num_embeddings, embedding_dim))

        params = layer.params()
        grads = layer.grads()
        self.assertIn("W", params)
        self.assertIn("W", grads)


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_forward(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4

        layer = LayerNorm(hidden_dim)
        X = np.random.randn(batch_size, seq_len, hidden_dim)

        out = layer(X)
        self.assertEqual(out.shape, (batch_size, seq_len, hidden_dim))

        X_norm = layer.cache['X_norm']
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


class TestSoftmax(unittest.TestCase):
    def test_softmax_forward(self):
        batch_size = 2
        num_classes = 5

        layer = Softmax()
        X = np.random.randn(batch_size, num_classes)

        out = layer(X)
        self.assertEqual(out.shape, (batch_size, num_classes))
        self.assertTrue(np.allclose(np.sum(out, axis=-1), 1.0))
        self.assertTrue(np.all(out >= 0))

    def test_softmax_backward(self):
        batch_size = 3
        num_classes = 4

        layer = Softmax()
        X = np.random.randn(batch_size, num_classes)
        Y_true = np.array([0, 2, 1])

        Y_hat = layer(X)
        dX = layer.backward(Y_hat, Y_true)

        self.assertEqual(dX.shape, (batch_size, num_classes))

        Y_onehot = np.zeros_like(Y_hat)
        Y_onehot[np.arange(Y_true.size), Y_true] = 1
        expected_dX = Y_hat - Y_onehot
        self.assertTrue(np.allclose(dX, expected_dX))

        params = layer.params()
        grads = layer.grads()
        self.assertEqual(len(params), 0)
        self.assertEqual(len(grads), 0)


class TestReLU(unittest.TestCase):
    def test_relu_forward(self):
        batch_size = 2
        seq_len = 3
        d_model = 4

        layer = ReLU()
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])

        out = layer(X)
        self.assertEqual(out.shape, (2, 4))

        expected = np.maximum(0, X)
        self.assertTrue(np.allclose(out, expected))

    def test_relu_backward(self):
        batch_size = 2
        seq_len = 3
        d_model = 4

        layer = ReLU()
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])
        dZ = np.ones_like(X)

        layer(X)
        dX = layer.backward(dZ)

        self.assertEqual(dX.shape, X.shape)

        # ReLU gradient: 1 if x > 0, else 0
        expected_grad = (X > 0).astype(np.float32)
        self.assertTrue(np.allclose(dX, expected_grad))

        params = layer.params()
        grads = layer.grads()
        self.assertEqual(len(params), 0)
        self.assertEqual(len(grads), 0)


class TestLeakyReLU(unittest.TestCase):
    def test_leaky_relu_forward(self):
        alpha = 0.01
        layer = LeakyReLU(alpha)
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])

        out = layer(X)
        self.assertEqual(out.shape, (2, 4))

        expected = np.where(X > 0, X, alpha * X)
        self.assertTrue(np.allclose(out, expected))

    def test_leaky_relu_backward(self):
        alpha = 0.01
        layer = LeakyReLU(alpha)
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])
        dZ = np.ones_like(X)

        layer(X)
        dX = layer.backward(dZ)

        self.assertEqual(dX.shape, X.shape)

        # LeakyReLU gradient: 1 if x > 0, else alpha
        expected_grad = np.where(X > 0, 1.0, alpha)
        self.assertTrue(np.allclose(dX, expected_grad))

        params = layer.params()
        grads = layer.grads()
        self.assertEqual(len(params), 0)
        self.assertEqual(len(grads), 0)


if __name__ == '__main__':
    unittest.main()
