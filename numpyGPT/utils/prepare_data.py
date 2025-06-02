import argparse
import os
import pickle

import numpy as np

from numpyGPT.tokenizer.bpe import BPETokenizer
from numpyGPT.tokenizer.char_level import CharTokenizer
from numpyGPT.tokenizer.word_level import WordTokenizer


def prepare_data(input_file, output_dir, tokenizer_type='char', train_split=0.9,
                 min_freq=1, max_vocab_size=None, vocab_size=1000):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Data length: {len(text):,} characters")

    if tokenizer_type == 'char':
        tokenizer = CharTokenizer()
    elif tokenizer_type == 'word':
        tokenizer = WordTokenizer(
            min_freq=min_freq,
            max_vocab_size=max_vocab_size
        )
    elif tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(
            vocab_size=vocab_size
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer.build_vocab(text)

    print(f"Vocab size: {tokenizer.vocab_size}")

    encoded = tokenizer.encode(text)
    data = np.array(encoded, dtype=np.uint16)

    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")

    os.makedirs(output_dir, exist_ok=True)

    train_data.tofile(os.path.join(output_dir, 'train.bin'))
    val_data.tofile(os.path.join(output_dir, 'val.bin'))

    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Done.")
    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_dir')
    parser.add_argument('--tokenizer_type', choices=['char', 'word', 'bpe'], default='char')
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--max_vocab_size', type=int, default=None)
    parser.add_argument('--vocab_size', type=int, default=1000)
    args = parser.parse_args()

    prepare_data(
        args.input_file,
        args.output_dir,
        args.tokenizer_type,
        args.train_split,
        args.min_freq,
        args.max_vocab_size,
        args.vocab_size
    )
