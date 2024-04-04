import numpy as np

from variable import Variable


class Function:
    def __call__(self, inp: Variable):
        x = inp.data
        y = self.forward(x)
        output = Variable(y)
        output.creator = self
        self.output = output
        self.input = inp
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, gy: np.ndarray):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.input.data) * gy


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

