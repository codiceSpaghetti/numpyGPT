import unittest

import numpy as np

from numpyGPT.nn.modules.activation import LeakyReLU, ReLU, Softmax


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
        expected_dX = (Y_hat - Y_onehot) / batch_size
        self.assertTrue(np.allclose(dX, expected_dX))

        params = layer.params()
        grads = layer.grads()
        self.assertEqual(len(params), 0)
        self.assertEqual(len(grads), 0)

    def test_softmax_general_backward(self):
        batch_size = 2
        seq_len = 4
        vocab_size = 5

        softmax = Softmax()
        X = np.random.randn(batch_size, seq_len, vocab_size)
        dZ = np.random.randn(batch_size, seq_len, vocab_size)

        X_reshaped = X.reshape(-1, vocab_size)
        dZ_reshaped = dZ.reshape(-1, vocab_size)

        probs = softmax(X_reshaped)
        dX = softmax.backward(dZ_reshaped)

        self.assertEqual(dX.shape, X_reshaped.shape)

    def test_softmax_classification_backward(self):
        batch_size = 3
        num_classes = 4

        softmax = Softmax()
        X = np.random.randn(batch_size, num_classes)
        Y_true = np.array([0, 2, 1])

        Y_hat = softmax(X)
        dX = softmax.backward(Y_hat, Y_true)

        self.assertEqual(dX.shape, (batch_size, num_classes))

        Y_onehot = np.zeros_like(Y_hat)
        Y_onehot[np.arange(Y_true.size), Y_true] = 1
        expected_dX = (Y_hat - Y_onehot) / batch_size
        self.assertTrue(np.allclose(dX, expected_dX))


if __name__ == '__main__':
    unittest.main()


