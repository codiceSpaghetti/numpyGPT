import unittest

import numpy as np

from numpyGPT.nn.modules.attention import MultiHeadAttention


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


if __name__ == '__main__':
    unittest.main()


