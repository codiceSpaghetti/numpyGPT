#! /usr/bin/env python3

import argparse
import os
import pickle

import numpy as np

from numpyGPT.models.GPT import GPT
from numpyGPT.utils.training import setup_logger


def sample_from_model(model_path, data_dir, num_samples=1, max_new_tokens=500,
                      temperature=0.8, start_text="\n", seed=1337):
    logger = setup_logger('sample')
    np.random.seed(seed)

    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = checkpoint['config']
    model = GPT(**config)

    model_params = model.params()
    for name, param in checkpoint['model'].items():
        model_params[name][:] = param

    logger.info("Model loaded")

    tokenizer_path = os.path.join(data_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    start_ids = tokenizer.encode(start_text)
    x = np.array(start_ids, dtype=np.int64)[None, ...]

    eos_token_id = tokenizer.eos_token_id

    logger.info(f"Generating {num_samples} samples...")

    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, eos_token_id=eos_token_id)

        print("\n" + "="*60)
        print(f"SAMPLE {k+1}/{num_samples}")
        print("="*60)
        decoded_text = tokenizer.decode(y[0].tolist())
        print(decoded_text)
        print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='out/ckpt.pkl')
    parser.add_argument('--data_dir', default='data/shakespeare_char_tokenized')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--start', default="\n")
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")

    if not os.path.exists(args.model_path):
        print(f"Model checkpoint not found: {args.model_path}")
        exit(1)

    sample_from_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        start_text=args.start,
        seed=args.seed
    )
