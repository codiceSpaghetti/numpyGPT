import os
import pickle

import numpy as np


class DataLoader:
    def __init__(self, data_dir, split, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        data_bin = os.path.join(data_dir, f'{split}.bin')
        self.data = np.fromfile(data_bin, dtype=np.uint16)

        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        self.stoi = meta['stoi']
        self.itos = meta['itos']

        self.current_pos = 0

    def get_batch(self):
        data = self.data
        ix = np.random.randint(0, len(data) - self.block_size, (self.batch_size,))
        X = np.stack([data[i:i+self.block_size] for i in ix])
        y = np.stack([data[i+1:i+self.block_size+1] for i in ix])
        return X.astype(np.int64), y.astype(np.int64)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
