import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, modules, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(modules, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = []
        self.v = []

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

    def step(self):
        self.t += 1

        for i, module in enumerate(self.params):
            params = module.params()
            grads = module.grads()

            for param_key in params:
                if grads[param_key] is None:
                    continue

                g = grads[param_key]

                self.m[i][param_key] = self.beta1 * self.m[i][param_key] + (1 - self.beta1) * g
                self.v[i][param_key] = self.beta2 * self.v[i][param_key] + (1 - self.beta2) * (g ** 2)

                m_hat = self.m[i][param_key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][param_key] / (1 - self.beta2 ** self.t)

                params[param_key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
