import argparse
import os
import pickle

import numpy as np

from numpyGPT.tokenizer.char_level import CharTokenizer


def prepare_data(input_file, output_dir, train_split=0.9):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Data length: {len(text):,} characters")

    tokenizer = CharTokenizer(special_tokens=['<pad>', '<unk>'])
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
    parser.add_argument('--train_split', type=float, default=0.9)
    args = parser.parse_args()

    prepare_data(args.input_file, args.output_dir, args.train_split)
