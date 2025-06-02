import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpyGPT.nn.modules.activation import ReLU, Softmax
from numpyGPT.nn.modules.attention import MultiHeadAttention
from numpyGPT.nn.modules.embedding import Embedding
from numpyGPT.nn.modules.feedforward import FeedForward
from numpyGPT.nn.modules.layerNorm import LayerNorm
from numpyGPT.nn.modules.linear import Linear
from numpyGPT.optim.adam import Adam


class TestComponents(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_linear(self):
        batch_size, in_features, out_features = 2, 4, 3

        np_linear = Linear(in_features, out_features)
        torch_linear = nn.Linear(in_features, out_features)

        torch_linear.weight.data = torch.from_numpy(np_linear.W.T).float()
        torch_linear.bias.data = torch.from_numpy(np_linear.b).float()

        x_np = np.random.randn(batch_size, in_features).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        out_np = np_linear(x_np)
        out_torch = torch_linear(x_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))

        grad_output = np.random.randn(batch_size, out_features).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_linear.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-5))
        self.assertTrue(np.allclose(np_linear.dW, torch_linear.weight.grad.T.numpy(), atol=1e-5))
        self.assertTrue(np.allclose(np_linear.db, torch_linear.bias.grad.numpy(), atol=1e-5))

    def test_embedding(self):
        batch_size, seq_len, vocab_size, embed_dim = 2, 5, 10, 8

        np_embedding = Embedding(vocab_size, embed_dim)
        torch_embedding = nn.Embedding(vocab_size, embed_dim)

        torch_embedding.weight.data = torch.from_numpy(np_embedding.W).float()

        indices = np.random.randint(0, vocab_size, (batch_size, seq_len))
        indices_torch = torch.from_numpy(indices).long()

        out_np = np_embedding(indices)
        out_torch = torch_embedding(indices_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))

        grad_output = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        np_embedding.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(np_embedding.dW, torch_embedding.weight.grad.numpy(), atol=1e-5))

    def test_layernorm(self):
        batch_size, seq_len, hidden_dim = 2, 5, 8

        np_layernorm = LayerNorm(hidden_dim)
        torch_layernorm = nn.LayerNorm(hidden_dim)

        torch_layernorm.weight.data = torch.from_numpy(np_layernorm.gamma).float()
        torch_layernorm.bias.data = torch.from_numpy(np_layernorm.beta).float()

        x_np = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        out_np = np_layernorm(x_np)
        out_torch = torch_layernorm(x_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-4))

        grad_output = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_layernorm.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-4))
        self.assertTrue(np.allclose(np_layernorm.dgamma, torch_layernorm.weight.grad.numpy(), atol=1e-4))
        self.assertTrue(np.allclose(np_layernorm.dbeta, torch_layernorm.bias.grad.numpy(), atol=1e-4))

    def test_relu(self):
        np_relu = ReLU()

        x_np = np.random.randn(3, 4).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        out_np = np_relu(x_np)
        out_torch = F.relu(x_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))

        grad_output = np.random.randn(3, 4).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_relu.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-5))

    def test_softmax(self):
        np_softmax = Softmax()

        x_np = np.random.randn(3, 5).astype(np.float32)
        y_true = np.array([0, 2, 1])

        x_torch = torch.from_numpy(x_np).requires_grad_(True)
        y_true_torch = torch.from_numpy(y_true).long()

        out_np = np_softmax(x_np)
        out_torch = F.softmax(x_torch, dim=-1)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))

        grad_input_np = np_softmax.backward(out_np, y_true)

        loss_torch = F.cross_entropy(x_torch, y_true_torch)
        loss_torch.backward()

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-5))

    def test_feedforward(self):
        batch_size, seq_len, d_model, d_ff = 2, 5, 8, 16

        np_ff = FeedForward(d_model, d_ff)

        class TorchFeedForward(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)

            def forward(self, x):
                return self.linear2(F.relu(self.linear1(x)))

        torch_ff = TorchFeedForward(d_model, d_ff)

        torch_ff.linear1.weight.data = torch.from_numpy(np_ff.linear1.W.T).float()
        torch_ff.linear1.bias.data = torch.from_numpy(np_ff.linear1.b).float()
        torch_ff.linear2.weight.data = torch.from_numpy(np_ff.linear2.W.T).float()
        torch_ff.linear2.bias.data = torch.from_numpy(np_ff.linear2.b).float()

        x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        out_np = np_ff(x_np)
        out_torch = torch_ff(x_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))

        grad_output = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_ff.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-5))

    def test_attention(self):
        batch_size, seq_len, d_model, n_heads = 2, 4, 8, 2

        np_mha = MultiHeadAttention(d_model, n_heads)
        torch_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        scale = 0.02
        np_mha.W_q.W = np.random.randn(d_model, d_model) * scale
        np_mha.W_k.W = np.random.randn(d_model, d_model) * scale
        np_mha.W_v.W = np.random.randn(d_model, d_model) * scale
        np_mha.W_o.W = np.random.randn(d_model, d_model) * scale

        qkv_weight = np.concatenate([
            np_mha.W_q.W.T, np_mha.W_k.W.T, np_mha.W_v.W.T
        ], axis=0)
        qkv_bias = np.concatenate([
            np_mha.W_q.b, np_mha.W_k.b, np_mha.W_v.b
        ])

        torch_mha.in_proj_weight.data = torch.from_numpy(qkv_weight).float()
        torch_mha.in_proj_bias.data = torch.from_numpy(qkv_bias).float()
        torch_mha.out_proj.weight.data = torch.from_numpy(np_mha.W_o.W.T).float()
        torch_mha.out_proj.bias.data = torch.from_numpy(np_mha.W_o.b).float()

        x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        mask_np = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        mask_torch = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

        out_np = np_mha(x_np, mask=mask_np)
        out_torch, _ = torch_mha(x_torch, x_torch, x_torch, attn_mask=mask_torch)

        self.assertTrue(np.allclose(out_np, out_torch.detach().numpy(), atol=1e-4))

        grad_output = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        grad_output_torch = torch.from_numpy(grad_output)

        grad_input_np = np_mha.backward(grad_output)
        out_torch.backward(grad_output_torch)

        self.assertTrue(np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-4))

    def test_adam(self):
        np_linear = Linear(4, 2)

        torch_linear = nn.Linear(4, 2)
        torch_linear.weight.data = torch.from_numpy(np_linear.W.T).float()
        torch_linear.bias.data = torch.from_numpy(np_linear.b).float()

        np_adam = Adam([np_linear], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        torch_adam = torch.optim.Adam(torch_linear.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        x = np.random.randn(3, 4).astype(np.float32)
        y_true = np.array([0, 1, 0])

        x_torch = torch.from_numpy(x)
        y_true_torch = torch.from_numpy(y_true).long()

        for step in range(3):
            out_np = np_linear(x)
            softmax = Softmax()
            probs_np = softmax(out_np)
            grad_np = softmax.backward(probs_np, y_true)
            np_linear.backward(grad_np)

            w_before_np = np_linear.W.copy()
            b_before_np = np_linear.b.copy()

            np_adam.step()

            torch_adam.zero_grad()
            out_torch = torch_linear(x_torch)
            loss_torch = F.cross_entropy(out_torch, y_true_torch)
            loss_torch.backward()

            w_before_torch = torch_linear.weight.data.clone()
            b_before_torch = torch_linear.bias.data.clone()

            torch_adam.step()

            w_update_np = np_linear.W - w_before_np
            b_update_np = np_linear.b - b_before_np
            w_update_torch = torch_linear.weight.data - w_before_torch
            b_update_torch = torch_linear.bias.data - b_before_torch

            self.assertTrue(np.allclose(w_update_np, w_update_torch.T.numpy(), atol=1e-6))
            self.assertTrue(np.allclose(b_update_np, b_update_torch.numpy(), atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
