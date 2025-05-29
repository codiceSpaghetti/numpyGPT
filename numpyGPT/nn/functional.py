import numpy as np


def cross_entropy_loss(Y_hat, Y_true, eps=1e-7):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_hat[np.arange(m), Y_true] + eps)
    return np.sum(log_likelihood) / m
