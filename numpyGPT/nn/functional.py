import numpy as np


def cross_entropy_loss(logits, targets, eps=1e-7):
    N = targets.shape[0]

    # logits -> log_softmax via log-sum-exp trick
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + eps)

    # negative log-likelihood: -log P(correct_class)
    nll = -log_softmax[np.arange(N), targets]

    return np.mean(nll)
