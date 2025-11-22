import unittest

import numpy as np

from numpyGPT.nn.modules.positional import PositionalEncoding


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


if __name__ == '__main__':
    unittest.main()


