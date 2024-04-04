import numpy as np


class Variable:
    def __init__(self, inp: np.ndarray):
        self.data = inp
        self.grad = None
        self.creator = None

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
