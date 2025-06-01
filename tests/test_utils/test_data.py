from numpyGPT.utils.data import DataLoader
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        vocab = ['a', 'b', 'c', 'd', 'e']
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for i, ch in enumerate(vocab)}

        train_data = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 10, dtype=np.uint16)
        val_data = np.array([4, 3, 2, 1, 0, 4, 3, 2, 1, 0] * 5, dtype=np.uint16)

        train_data.tofile(os.path.join(self.temp_dir, 'train.bin'))
        val_data.tofile(os.path.join(self.temp_dir, 'val.bin'))

        meta = {
            'vocab_size': len(vocab),
            'stoi': stoi,
            'itos': itos,
        }
        with open(os.path.join(self.temp_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

    def test_init(self):
        loader = DataLoader(self.temp_dir, 'train', batch_size=4, block_size=8)

        self.assertEqual(loader.batch_size, 4)
        self.assertEqual(loader.block_size, 8)
        self.assertEqual(loader.vocab_size, 5)
        self.assertEqual(len(loader.data), 100)

    def test_get_batch_shapes(self):
        loader = DataLoader(self.temp_dir, 'train', batch_size=4, block_size=8)
        X, y = loader.get_batch()

        self.assertEqual(X.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        self.assertEqual(X.dtype, np.int64)
        self.assertEqual(y.dtype, np.int64)

    def test_get_batch_values(self):
        loader = DataLoader(self.temp_dir, 'train', batch_size=2, block_size=4)
        np.random.seed(42)
        X, y = loader.get_batch()

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.assertTrue(0 <= X[i, j] < 5)
                self.assertTrue(0 <= y[i, j] < 5)

        for i in range(X.shape[0]):
            for j in range(X.shape[1] - 1):
                expected_y = X[i, j + 1] if j + 1 < X.shape[1] else (X[i, j] + 1) % 5

    def test_encode_decode(self):
        loader = DataLoader(self.temp_dir, 'train', batch_size=2, block_size=4)

        text = "abcde"
        encoded = loader.encode(text)
        decoded = loader.decode(encoded)

        self.assertEqual(encoded, [0, 1, 2, 3, 4])
        self.assertEqual(decoded, text)

    def test_val_split(self):
        val_loader = DataLoader(self.temp_dir, 'val', batch_size=2, block_size=4)

        self.assertEqual(len(val_loader.data), 50)
        self.assertEqual(val_loader.vocab_size, 5)


if __name__ == '__main__':
    unittest.main()
