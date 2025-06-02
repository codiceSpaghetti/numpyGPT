import unittest

import numpy as np

from numpyGPT.models.GPT import GPT
from numpyGPT.nn.modules.activation import LeakyReLU, ReLU, Softmax
from numpyGPT.nn.modules.attention import MultiHeadAttention
from numpyGPT.nn.modules.feedforward import FeedForward
from numpyGPT.nn.modules.positional import PositionalEncoding
from numpyGPT.nn.modules.transformer import TransformerBlock


class TestReLU(unittest.TestCase):
    def test_relu_forward(self):
        relu = ReLU()
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])

        out = relu(X)
        self.assertEqual(out.shape, X.shape)

        expected = np.maximum(0, X)
        self.assertTrue(np.allclose(out, expected))

    def test_relu_backward(self):
        relu = ReLU()
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])
        dZ = np.ones_like(X)

        relu(X)
        dX = relu.backward(dZ)

        self.assertEqual(dX.shape, X.shape)
        expected_grad = (X > 0).astype(np.float32)
        self.assertTrue(np.allclose(dX, expected_grad))


class TestLeakyReLU(unittest.TestCase):
    def test_leaky_relu_forward(self):
        alpha = 0.01
        leaky_relu = LeakyReLU(alpha)
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])

        out = leaky_relu(X)
        self.assertEqual(out.shape, X.shape)

        expected = np.where(X > 0, X, alpha * X)
        self.assertTrue(np.allclose(out, expected))

    def test_leaky_relu_backward(self):
        alpha = 0.01
        leaky_relu = LeakyReLU(alpha)
        X = np.array([[-1, 0, 1, 2], [3, -2, 0.5, -0.5]])
        dZ = np.ones_like(X)

        leaky_relu(X)
        dX = leaky_relu.backward(dZ)

        self.assertEqual(dX.shape, X.shape)
        expected_grad = np.where(X > 0, 1.0, alpha)
        self.assertTrue(np.allclose(dX, expected_grad))


class TestMultiHeadAttention(unittest.TestCase):
    def test_attention_forward(self):
        d_model = 64
        n_heads = 8
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(d_model, n_heads)
        X = np.random.randn(batch_size, seq_len, d_model)

        out = attn(X)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_attention_backward(self):
        d_model = 64
        n_heads = 8
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(d_model, n_heads)
        X = np.random.randn(batch_size, seq_len, d_model)
        dZ = np.random.randn(batch_size, seq_len, d_model)

        attn(X)
        dX = attn.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, seq_len, d_model))

        params = attn.params()
        grads = attn.grads()
        self.assertEqual(len(params), 8)
        self.assertEqual(len(grads), 8)

    def test_attention_with_mask(self):
        d_model = 64
        n_heads = 8
        batch_size = 2
        seq_len = 10

        attn = MultiHeadAttention(d_model, n_heads)
        X = np.random.randn(batch_size, seq_len, d_model)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        mask = mask[None, None, :, :]

        out = attn(X, mask)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))


class TestFeedForward(unittest.TestCase):
    def test_feedforward_forward(self):
        d_model = 64
        d_ff = 256
        batch_size = 2
        seq_len = 10

        ffn = FeedForward(d_model, d_ff)
        X = np.random.randn(batch_size, seq_len, d_model)

        out = ffn(X)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_feedforward_backward(self):
        d_model = 64
        d_ff = 256
        batch_size = 2
        seq_len = 10

        ffn = FeedForward(d_model, d_ff)
        X = np.random.randn(batch_size, seq_len, d_model)
        dZ = np.random.randn(batch_size, seq_len, d_model)

        ffn(X)
        dX = ffn.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, seq_len, d_model))

        params = ffn.params()
        grads = ffn.grads()
        self.assertEqual(len(params), 4)
        self.assertEqual(len(grads), 4)


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_forward(self):
        max_len = 100
        d_model = 64
        batch_size = 2
        seq_len = 10

        pos_enc = PositionalEncoding(max_len, d_model)
        X = np.random.randn(batch_size, seq_len, d_model)

        out = pos_enc(X)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_positional_backward(self):
        max_len = 100
        d_model = 64
        batch_size = 2
        seq_len = 10

        pos_enc = PositionalEncoding(max_len, d_model)
        X = np.random.randn(batch_size, seq_len, d_model)
        dZ = np.random.randn(batch_size, seq_len, d_model)

        pos_enc(X)
        dX = pos_enc.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, seq_len, d_model))

        params = pos_enc.params()
        grads = pos_enc.grads()
        self.assertIn("W", params)
        self.assertIn("W", grads)


class TestTransformerBlock(unittest.TestCase):
    def test_transformer_block_forward(self):
        d_model = 64
        n_heads = 8
        d_ff = 256
        batch_size = 2
        seq_len = 10

        block = TransformerBlock(d_model, n_heads, d_ff)
        X = np.random.randn(batch_size, seq_len, d_model)

        out = block(X)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_transformer_block_backward(self):
        d_model = 64
        n_heads = 8
        d_ff = 256
        batch_size = 2
        seq_len = 10

        block = TransformerBlock(d_model, n_heads, d_ff)
        X = np.random.randn(batch_size, seq_len, d_model)
        dZ = np.random.randn(batch_size, seq_len, d_model)

        block(X)
        dX = block.backward(dZ)

        self.assertEqual(dX.shape, (batch_size, seq_len, d_model))


class TestSoftmaxEnhanced(unittest.TestCase):
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


class TestGPT(unittest.TestCase):
    def test_gpt_forward(self):
        vocab_size = 100
        max_len = 32
        d_model = 64
        n_heads = 8
        n_layers = 2
        d_ff = 256

        model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)
        X = np.random.randint(0, vocab_size, size=(2, 16))

        logits = model(X)
        self.assertEqual(logits.shape, (2, 16, vocab_size))

    def test_gpt_forward_with_targets(self):
        vocab_size = 100
        max_len = 32
        d_model = 64
        n_heads = 8
        n_layers = 2
        d_ff = 256

        model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)
        X = np.random.randint(0, vocab_size, size=(2, 16))
        targets = np.random.randint(0, vocab_size, size=(2, 16))

        logits, loss = model(X, targets)
        self.assertEqual(logits.shape, (2, 16, vocab_size))
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)

    def test_gpt_backward(self):
        vocab_size = 50
        max_len = 16
        d_model = 32
        n_heads = 4
        n_layers = 2
        d_ff = 128

        model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)
        X = np.random.randint(0, vocab_size, size=(2, 8))
        targets = np.random.randint(0, vocab_size, size=(2, 8))

        logits, loss = model(X, targets)
        model.backward()

        grads = model.grads()
        self.assertGreater(len(grads), 0)

        for grad_name, grad_value in grads.items():
            self.assertIsNotNone(grad_value)

    def test_gpt_generate(self):
        vocab_size = 50
        max_len = 16
        d_model = 32
        n_heads = 4
        n_layers = 2
        d_ff = 128

        model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)
        start_tokens = np.array([[1, 2, 3]])

        generated = model.generate(start_tokens, max_new_tokens=5)
        self.assertEqual(generated.shape, (1, 8))
        self.assertTrue(np.array_equal(generated[:, :3], start_tokens))

    def test_gpt_params_and_grads(self):
        vocab_size = 30
        max_len = 8
        d_model = 16
        n_heads = 2
        n_layers = 1
        d_ff = 64

        model = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)
        X = np.random.randint(0, vocab_size, size=(1, 4))
        targets = np.random.randint(0, vocab_size, size=(1, 4))

        logits, loss = model(X, targets)
        model.backward()

        params = model.params()
        grads = model.grads()

        self.assertEqual(len(params), len(grads))
        for param_name in params.keys():
            self.assertIn(param_name, grads)


if __name__ == '__main__':
    unittest.main()
