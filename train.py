#! /usr/bin/env python3

import json
import os
import pickle
import time

import numpy as np

from numpyGPT.models.GPT import GPT
from numpyGPT.optim.adam import Adam
from numpyGPT.optim.lr_scheduler.warmup_cosine_lr import WarmupCosineLR
from numpyGPT.utils.data.dataloader import DataLoader
from numpyGPT.utils.training import (
    TrainingMonitor,
    clip_grad_norm,
    get_lr,
    setup_logger,
)
from numpyGPT.utils.vis import MetricsLogger

data_dir = 'data/shakespeare_char'
out_dir = 'out/char'
eval_interval = 250
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
resume = True

batch_size = 16
block_size = 128
max_iters = 8000
lr = 3e-4
min_lr = 3e-5

n_layer = 4
n_head = 4
n_embd = 256

warmup_iters = 800
lr_decay_iters = 8000
grad_clip = 1.0

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

logger = setup_logger('train')

train_loader = DataLoader(data_dir, 'train', batch_size, block_size)
val_loader = DataLoader(data_dir, 'val', batch_size, block_size)

vocab_size = train_loader.vocab_size
logger.info(f"vocab_size: {vocab_size}")

model = GPT(vocab_size=vocab_size, max_len=block_size, d_model=n_embd,
            n_heads=n_head, n_layers=n_layer, d_ff=4*n_embd)

optimizer = Adam([model], lr=lr)
scheduler = WarmupCosineLR(optimizer, warmup_iters, lr_decay_iters, min_lr)
monitor = TrainingMonitor(log_interval)
metrics = MetricsLogger(os.path.join(out_dir, 'metrics.json'))


def save_model(filepath, model, iter_num, val_loss=None, optimizer_state=None):
    model_data = {
        'model': model.params(),
        'iter_num': iter_num,
        'config': {
            'vocab_size': vocab_size,
            'max_len': block_size,
            'd_model': n_embd,
            'n_heads': n_head,
            'n_layers': n_layer,
            'd_ff': 4*n_embd,
        }
    }

    if val_loss is not None:
        model_data['val_loss'] = val_loss

    if optimizer_state is not None:
        model_data['optimizer_state'] = optimizer_state
        model_data['best_val_loss'] = best_val_loss

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        loader = train_loader if split == 'train' else val_loader
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


iter_num = 0
best_val_loss = 1e9
resume_from_checkpoint = False

ckpt_path = os.path.join(out_dir, 'ckpt.pkl')
if resume and os.path.exists(ckpt_path):
    logger.info(f"resuming training from {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        checkpoint = pickle.load(f)

    model_params = model.params()
    for name, param in checkpoint['model'].items():
        model_params[name][:] = param

    if 'optimizer_state' in checkpoint:
        optimizer.m = checkpoint['optimizer_state']['m']
        optimizer.v = checkpoint['optimizer_state']['v']
        optimizer.t = checkpoint['optimizer_state']['t']
        logger.info("restored optimizer state")

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    for _ in range(iter_num):
        scheduler.step()

    resume_from_checkpoint = True
    logger.info(f"resumed from iteration {iter_num}, best_val_loss={best_val_loss:.4f}")

num_params = sum(p.size for p in model.params().values())
logger.info(f"number of parameters: {num_params/1e6:.2f}M")

if not resume_from_checkpoint:
    logger.info("starting training from scratch")

t0 = time.time()

while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        metrics.log(iter_num, val_loss=losses['val'], lr=get_lr(optimizer))

        if losses['val'] < best_val_loss or always_save_checkpoint:
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                should_save_best = True
            else:
                should_save_best = False

            if iter_num > 0:
                optimizer_state = {
                    'm': optimizer.m,
                    'v': optimizer.v,
                    't': optimizer.t
                }
                logger.info(f"saving checkpoint to {out_dir}")
                save_model(os.path.join(out_dir, 'ckpt.pkl'), model, iter_num,
                           optimizer_state=optimizer_state)

                if should_save_best:
                    save_model(os.path.join(out_dir, 'best_model.pkl'), model, iter_num,
                               val_loss=losses['val'])

    optimizer.zero_grad()

    X, Y = train_loader.get_batch()

    t1 = time.time()
    logits, loss = model(X, Y)
    model.backward()

    if grad_clip != 0.0:
        grad_norm = clip_grad_norm(model, grad_clip)
    else:
        grad_norm = None

    optimizer.step()
    scheduler.step()

    t2 = time.time()
    dt = t2 - t1

    metrics.log(iter_num, train_loss=loss, grad_norm=grad_norm, lr=get_lr(optimizer))

    log_msg = monitor.log_step(iter_num, loss, get_lr(optimizer), grad_norm)
    if log_msg:
        logger.info(log_msg)

    iter_num += 1

    if iter_num > max_iters:
        break

t1 = time.time()
dt = t1 - t0
logger.info(f"training finished in {dt:.2f}s")

plot_path = metrics.plot(os.path.join(out_dir, 'training_curves.png'))
logger.info(f"training curves saved to {plot_path}")
