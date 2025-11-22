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
    Inspired by GPT-2 architecture: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.max_len: int = max_len
        self.d_model: int = d_model

        self.tok_emb: Embedding = Embedding(vocab_size, d_model)
        self.pos_emb: PositionalEncoding = PositionalEncoding(max_len, d_model)
        self.blocks: list[TransformerBlock] = [
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]
        self.ln_f: LayerNorm = LayerNorm(d_model)
        self.lm_head: Linear = Linear(d_model, vocab_size)

        self.cache: dict[str, np.ndarray] = {}

    def forward(
        self, X: np.ndarray, targets: np.ndarray | None = None
    ) -> np.ndarray | tuple[np.ndarray, float]:
        B, T = X.shape

        tok_emb = self.tok_emb(X)  # (B, T, C)
        X = self.pos_emb(tok_emb)  # (B, T, C)

        mask = self._create_causal_mask(T)  # (1, 1, T, T)

        for block in self.blocks:
            X = block(X, mask)  # (B, T, C)

        X = self.ln_f(X)  # (B, T, C)
        logits = self.lm_head(X)  # (B, T, vocab_size)

        if targets is not None:
            loss = cross_entropy_loss(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.cache = {"logits": logits, "targets": targets}
            return logits, loss

        return logits

    def backward(self) -> None:
        logits, targets = self.cache["logits"], self.cache["targets"]
        B, T, C = logits.shape

        logits_reshaped = logits.reshape(-1, C)  # (B*T, C)
        targets_reshaped = targets.reshape(-1)  # (B*T)

        logits_max = np.max(logits_reshaped, axis=-1, keepdims=True)  # (B*T, 1)
        exp_logits = np.exp(logits_reshaped - logits_max)  # for numerically stable softmax
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (B*T, C)

        # CE + softmax trick (https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy)
        dlogits = probs
        dlogits[np.arange(len(targets_reshaped)), targets_reshaped] -= (
            1  # d(CE)/d(logits) = softmax - one_hot
        )
        dlogits /= len(targets_reshaped)  # average over batch and sequence length

        dlogits = dlogits.reshape(B, T, C)

        dX = self.lm_head.backward(dlogits)
        dX = self.ln_f.backward(dX)

        for block in reversed(self.blocks):
            dX = block.backward(dX)

        dX = self.pos_emb.backward(dX)
        self.tok_emb.backward(dX)

    def _create_causal_mask(self, T: int) -> np.ndarray:
        mask = (
            np.triu(np.ones((T, T)), k=1) * -1e9
        )  # will add a large negative value to the scores of future tokens
        return mask[None, None, :, :]

    def generate(
        self,
        idx: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> np.ndarray:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            logits_shifted = logits - np.max(
                logits, axis=-1, keepdims=True
            )  # for numerically stable softmax
            probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=-1, keepdims=True)

            idx_next = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])
            idx = np.concatenate([idx, idx_next], axis=1)

            if eos_token_id is not None and idx_next[0, 0] == eos_token_id:
                break

        return idx

    def params(self) -> dict[str, np.ndarray]:
        params = {}
        params.update({f"tok_emb.{k}": v for k, v in self.tok_emb.params().items()})
        params.update({f"pos_emb.{k}": v for k, v in self.pos_emb.params().items()})

        for i, block in enumerate(self.blocks):
            params.update({f"blocks.{i}.{k}": v for k, v in block.params().items()})

        params.update({f"ln_f.{k}": v for k, v in self.ln_f.params().items()})
        params.update({f"lm_head.{k}": v for k, v in self.lm_head.params().items()})
        return params

    def grads(self) -> dict[str, np.ndarray | None]:
        grads = {}
        grads.update({f"tok_emb.{k}": v for k, v in self.tok_emb.grads().items()})
        grads.update({f"pos_emb.{k}": v for k, v in self.pos_emb.grads().items()})

        for i, block in enumerate(self.blocks):
            grads.update({f"blocks.{i}.{k}": v for k, v in block.grads().items()})

        grads.update({f"ln_f.{k}": v for k, v in self.ln_f.grads().items()})
        grads.update({f"lm_head.{k}": v for k, v in self.lm_head.grads().items()})
        return grads
