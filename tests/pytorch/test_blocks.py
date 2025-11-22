import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpyGPT.models.GPT import GPT
from numpyGPT.nn.modules.positional import PositionalEncoding
from numpyGPT.nn.modules.transformer import TransformerBlock


class TestBlocks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_positional_encoding(self):
        batch_size, seq_len, d_model = 2, 10, 8

        np_pos_enc = PositionalEncoding(seq_len, d_model)

        class TorchPositionalEncoding(nn.Module):
            def __init__(self, max_len, d_model):
                super().__init__()
                pe = torch.from_numpy(np_pos_enc.W).float()
                self.register_buffer("pe", pe)

            def forward(self, x):
                T = x.size(1)
                return x + self.pe[:T, :].unsqueeze(0)

        torch_pos_enc = TorchPositionalEncoding(seq_len, d_model)

        x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        out_np = np_pos_enc(x_np)
        out_torch = torch_pos_enc(x_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-4))

        grad_output = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_pos_enc.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-4))

    def test_transformer_block(self):
        batch_size, seq_len, d_model, n_heads, d_ff = 2, 6, 8, 2, 16

        np_block = TransformerBlock(d_model, n_heads, d_ff)

        scale = 0.02
        np_block.attn.W_q.W = np.random.randn(d_model, d_model) * scale
        np_block.attn.W_k.W = np.random.randn(d_model, d_model) * scale
        np_block.attn.W_v.W = np.random.randn(d_model, d_model) * scale
        np_block.attn.W_o.W = np.random.randn(d_model, d_model) * scale
        np_block.ffn.linear1.W = np.random.randn(d_model, d_ff) * scale
        np_block.ffn.linear2.W = np.random.randn(d_ff, d_model) * scale

        class TorchTransformerBlock(nn.Module):
            def __init__(self, d_model, n_heads, d_ff):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.feed_forward = nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
                )
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x, mask=None):
                norm1_out = self.norm1(x)
                attn_out, _ = self.self_attn(norm1_out, norm1_out, norm1_out, attn_mask=mask)
                x = x + attn_out

                norm2_out = self.norm2(x)
                ff_out = self.feed_forward(norm2_out)
                x = x + ff_out

                return x

        torch_block = TorchTransformerBlock(d_model, n_heads, d_ff)

        qkv_weight = np.concatenate(
            [np_block.attn.W_q.W.T, np_block.attn.W_k.W.T, np_block.attn.W_v.W.T], axis=0
        )
        qkv_bias = np.concatenate([np_block.attn.W_q.b, np_block.attn.W_k.b, np_block.attn.W_v.b])

        torch_block.self_attn.in_proj_weight.data = torch.from_numpy(qkv_weight).float()
        torch_block.self_attn.in_proj_bias.data = torch.from_numpy(qkv_bias).float()
        torch_block.self_attn.out_proj.weight.data = torch.from_numpy(np_block.attn.W_o.W.T).float()
        torch_block.self_attn.out_proj.bias.data = torch.from_numpy(np_block.attn.W_o.b).float()

        torch_block.feed_forward[0].weight.data = torch.from_numpy(np_block.ffn.linear1.W.T).float()
        torch_block.feed_forward[0].bias.data = torch.from_numpy(np_block.ffn.linear1.b).float()
        torch_block.feed_forward[2].weight.data = torch.from_numpy(np_block.ffn.linear2.W.T).float()
        torch_block.feed_forward[2].bias.data = torch.from_numpy(np_block.ffn.linear2.b).float()

        torch_block.norm1.weight.data = torch.from_numpy(np_block.ln1.gamma).float()
        torch_block.norm1.bias.data = torch.from_numpy(np_block.ln1.beta).float()
        torch_block.norm2.weight.data = torch.from_numpy(np_block.ln2.gamma).float()
        torch_block.norm2.bias.data = torch.from_numpy(np_block.ln2.beta).float()

        x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        mask_np = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        mask_torch = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)

        out_np = np_block(x_np, mask=mask_np)
        out_torch = torch_block(x_torch, mask=mask_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-4))

        grad_output = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_block.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-4))

    def test_gpt_forward(self):
        vocab_size, max_len, d_model, n_heads, n_layers, d_ff = 50, 16, 16, 4, 2, 64

        np_gpt = GPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)

        class TorchGPT(nn.Module):
            def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff):
                super().__init__()
                self.token_embedding_table = nn.Embedding(vocab_size, d_model)
                self.position_embedding_table = nn.Embedding(max_len, d_model)
                self.blocks = nn.Sequential(
                    *[
                        nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=n_heads,
                            dim_feedforward=d_ff,
                            activation="relu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(n_layers)
                    ]
                )
                self.ln_f = nn.LayerNorm(d_model)
                self.lm_head = nn.Linear(d_model, vocab_size)

            def forward(self, idx):
                B, T = idx.shape
                tok_emb = self.token_embedding_table(idx)
                pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
                x = tok_emb + pos_emb

                mask = torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)

                for block in self.blocks:
                    x = block(x, src_mask=mask)

                x = self.ln_f(x)
                logits = self.lm_head(x)
                return logits

        torch_gpt = TorchGPT(vocab_size, max_len, d_model, n_heads, n_layers, d_ff)

        batch_size, seq_len = 2, 8
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        input_ids_torch = torch.from_numpy(input_ids).long()

        out_np = np_gpt(input_ids)
        out_torch = torch_gpt(input_ids_torch)

        self.assertEqual(out_np.shape, out_torch.shape)
        self.assertEqual(out_np.shape, (batch_size, seq_len, vocab_size))

    def test_cross_entropy_loss(self):
        batch_size, seq_len, vocab_size = 2, 8, 50

        logits_np = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        logits_torch = torch.from_numpy(logits_np).requires_grad_(True)
        targets_torch = torch.from_numpy(targets).long()

        from numpyGPT.nn.modules.activation import Softmax

        softmax = Softmax()

        logits_reshaped = logits_np.reshape(-1, vocab_size)
        targets_reshaped = targets.reshape(-1)

        probs = softmax(logits_reshaped)

        loss_np = -np.mean(np.log(probs[np.arange(len(targets_reshaped)), targets_reshaped] + 1e-8))

        grad_logits_reshaped = softmax.backward(probs, targets_reshaped)
        grad_logits_np = grad_logits_reshaped.reshape(batch_size, seq_len, vocab_size)

        loss_torch = F.cross_entropy(logits_torch.view(-1, vocab_size), targets_torch.view(-1))
        loss_torch.backward()

        self.assertTrue(abs(loss_np - loss_torch.item()) < 1e-6)
        self.assertTrue(np.allclose(grad_logits_np, logits_torch.grad.numpy(), atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
