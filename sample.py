#! /usr/bin/env python3

import os
import pickle

import numpy as np

from numpyGPT.models.GPT import GPT
from numpyGPT.utils.data.dataloader import DataLoader
from numpyGPT.utils.training import setup_logger

out_dir = 'out'
data_dir = 'data/shakespeare_'
num_samples = 1
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
start = "\n"

logger = setup_logger('sample')
np.random.seed(seed)

ckpt_path = os.path.join(out_dir, 'ckpt.pkl')
logger.info(f"loading model from {ckpt_path}")

with open(ckpt_path, 'rb') as f:
    checkpoint = pickle.load(f)

config = checkpoint['config']
logger.info(f"model config: {config}")

model = GPT(**config)

model_params = model.params()
for name, param in checkpoint['model'].items():
    model_params[name][:] = param

logger.info(f"model loaded successfully")

train_loader = DataLoader(data_dir, 'train', 1, 1)

start_ids = train_loader.encode(start)
x = np.array(start_ids, dtype=np.int64)[None, ...]

logger.info(f"generating {num_samples} samples with {max_new_tokens} tokens each")
logger.info(f"temperature: {temperature}, start: '{start}'")

for k in range(num_samples):
    logger.info(f"generating sample {k+1}/{num_samples}...")
    y = model.generate(x, max_new_tokens, temperature=temperature)

    print("\n" + "="*50)
    print(f"SAMPLE {k+1}")
    print("="*50)
    print(train_loader.decode(y[0].tolist()))
    print("="*50 + "\n")
