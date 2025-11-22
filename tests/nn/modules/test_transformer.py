import unittest

import numpy as np

from numpyGPT.nn.modules.transformer import TransformerBlock


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


if __name__ == '__main__':
    unittest.main()


