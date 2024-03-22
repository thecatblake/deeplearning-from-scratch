import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    c = np.max(x)
    exp = np.exp(x - c)
    sum_exp = np.sum(exp)
    y = exp / sum_exp
    return y
