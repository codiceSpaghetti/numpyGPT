from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
