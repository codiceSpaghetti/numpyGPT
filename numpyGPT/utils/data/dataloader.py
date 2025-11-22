import os
import pickle

import numpy as np
from numpy import ndarray

from ...tokenizer import Tokenizer


class DataLoader:
    def __init__(self, data_dir: str, split: str, batch_size: int, block_size: int) -> None:
        self.batch_size: int = batch_size
        self.block_size: int = block_size

        data_bin = os.path.join(data_dir, f"{split}.bin")
        self.data: ndarray = np.fromfile(data_bin, dtype=np.uint16)

        tokenizer_path = os.path.join(data_dir, "tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            self.tokenizer: Tokenizer = pickle.load(f)

        self.vocab_size: int = self.tokenizer.vocab_size
        self.current_pos: int = 0

    def get_batch(self) -> tuple[ndarray, ndarray]:
        data = self.data
        ix = np.random.randint(0, len(data) - self.block_size, (self.batch_size,))
        X = np.stack([data[i : i + self.block_size] for i in ix])
        y = np.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return X.astype(np.int64), y.astype(np.int64)

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
