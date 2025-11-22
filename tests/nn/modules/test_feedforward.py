import unittest

import numpy as np

from numpyGPT.nn.modules.feedforward import FeedForward


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


if __name__ == "__main__":
    unittest.main()
