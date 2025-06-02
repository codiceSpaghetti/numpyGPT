import os
import pickle

import numpy as np


class DataLoader:
    def __init__(self, data_dir, split, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        data_bin = os.path.join(data_dir, f'{split}.bin')
        self.data = np.fromfile(data_bin, dtype=np.uint16)

        tokenizer_path = os.path.join(data_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.vocab_size = self.tokenizer.vocab_size
        self.current_pos = 0

    def get_batch(self):
        data = self.data
        ix = np.random.randint(0, len(data) - self.block_size, (self.batch_size,))
        X = np.stack([data[i:i+self.block_size] for i in ix])
        y = np.stack([data[i+1:i+self.block_size+1] for i in ix])
        return X.astype(np.int64), y.astype(np.int64)

    def encode(self, s):
        return self.tokenizer.encode(s)

    def decode(self, l):
        return self.tokenizer.decode(l)
