import numpy as np

from ..nn.functional import cross_entropy_loss
from ..nn.modules.embedding import Embedding
from ..nn.modules.layerNorm import LayerNorm
from ..nn.modules.linear import Linear
from ..nn.modules.module import Module
from ..nn.modules.positional import PositionalEncoding
from ..nn.modules.transformer import TransformerBlock


class GPT(Module):
    """
    Inspiried by GPT-2 architecture: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        self.cache = {}

    def forward(self, X, targets=None):
        B, T = X.shape  # (B, T)

        tok_emb = self.tok_emb(X)  # (B, T, C)
        X = self.pos_emb(tok_emb)  # (B, T, C)

        mask = self._create_causal_mask(T)  # (1, 1, T, T)

        for block in self.blocks:
            X = block(X, mask)  # (B, T, C)

        X = self.ln_f(X)  # (B, T, C)
        logits = self.lm_head(X)  # (B, T, vocab_size)

        if targets is not None:
            loss = cross_entropy_loss(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.cache = {'logits': logits, 'targets': targets}
            return logits, loss

        return logits

    def backward(self):
        logits, targets = self.cache['logits'], self.cache['targets']
        B, T, C = logits.shape

        # d(CE)/d(logits) = softmax - one_hot
        logits_reshaped = logits.reshape(-1, C)
        targets_reshaped = targets.reshape(-1)

        logits_max = np.max(logits_reshaped, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_reshaped - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        dlogits = probs
        dlogits[np.arange(len(targets_reshaped)), targets_reshaped] -= 1
        dlogits /= len(targets_reshaped)

        dlogits = dlogits.reshape(B, T, C)

        dX = self.lm_head.backward(dlogits)
        dX = self.ln_f.backward(dX)

        for block in reversed(self.blocks):
            dX = block.backward(dX)

        dX = self.pos_emb.backward(dX)
        self.tok_emb.backward(dX)

    def _create_causal_mask(self, T):
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        return mask[None, None, :, :]

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=-1, keepdims=True)

            idx_next = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])
            idx = np.concatenate([idx, idx_next], axis=1)

        return idx

    def params(self):
        params = {}
        params.update({f"tok_emb.{k}": v for k, v in self.tok_emb.params().items()})
        params.update({f"pos_emb.{k}": v for k, v in self.pos_emb.params().items()})

        for i, block in enumerate(self.blocks):
            params.update({f"blocks.{i}.{k}": v for k, v in block.params().items()})

        params.update({f"ln_f.{k}": v for k, v in self.ln_f.params().items()})
        params.update({f"lm_head.{k}": v for k, v in self.lm_head.params().items()})
        return params

    def grads(self):
        grads = {}
        grads.update({f"tok_emb.{k}": v for k, v in self.tok_emb.grads().items()})
        grads.update({f"pos_emb.{k}": v for k, v in self.pos_emb.grads().items()})

        for i, block in enumerate(self.blocks):
            grads.update({f"blocks.{i}.{k}": v for k, v in block.grads().items()})

        grads.update({f"ln_f.{k}": v for k, v in self.ln_f.grads().items()})
        grads.update({f"lm_head.{k}": v for k, v in self.lm_head.grads().items()})
        return grads
