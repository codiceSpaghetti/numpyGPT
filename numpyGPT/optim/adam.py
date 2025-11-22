import numpy as np
from numpy import ndarray

from ..nn.modules.module import Module
from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer: https://arxiv.org/abs/1412.6980
    """

    def __init__(self, modules: list[Module], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        super().__init__(modules, lr)
        self.beta1: float
        self.beta2: float
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        self.t: int = 0

        self.m: list[dict[str, ndarray]] = []
        self.v: list[dict[str, ndarray]] = []

        for module in self.params:
            m_dict = {}
            v_dict = {}
            params = module.params()
            for param_key in params:
                param_shape = params[param_key].shape
                m_dict[param_key] = np.zeros(param_shape)
                v_dict[param_key] = np.zeros(param_shape)
            self.m.append(m_dict)
            self.v.append(v_dict)

    def step(self) -> None:
        self.t += 1

        for i, module in enumerate(self.params):
            params = module.params()
            grads = module.grads()

            for param_key in params:
                if grads[param_key] is None:
                    continue

                g = grads[param_key]

                self.m[i][param_key] = self.beta1 * self.m[i][param_key] + (1 - self.beta1) * g         # m_t   = β1 * m_{t-1} + (1 - β1) * g_t
                self.v[i][param_key] = self.beta2 * self.v[i][param_key] + (1 - self.beta2) * (g ** 2)  # v_t   = β2 * v_{t-1} + (1 - β2) * g_t^2

                m_hat = self.m[i][param_key] / (1 - self.beta1 ** self.t)  # m̂_t  = m_t / (1 - β1^t)
                v_hat = self.v[i][param_key] / (1 - self.beta2 ** self.t)  # v̂_t  = v_t / (1 - β2^t)

                params[param_key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + eps)
