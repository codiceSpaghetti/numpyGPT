import unittest

import numpy as np

from numpyGPT.nn.modules.embedding import Embedding


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


if __name__ == '__main__':
    unittest.main()


