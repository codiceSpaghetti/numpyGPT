class Optimizer:
    def __init__(self, modules, lr=0.001):
        if self.__class__ == Optimizer:
            raise TypeError("Cannot instantiate abstract class Optimizer directly")
        self.params = modules
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for module in self.params:
            grads = module.grads()
            for grad_key in grads:
                if grads[grad_key] is not None:
                    grads[grad_key].fill(0.0)
