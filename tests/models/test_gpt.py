import unittest

import numpy as np

from numpyGPT.models.GPT import GPT


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


