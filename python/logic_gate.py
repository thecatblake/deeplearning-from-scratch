import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    out = np.sum(x * w) + b
    if out <= 0:
        return 0
    return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    out = np.sum(x * w) + b
    if out <= 0:
        return 0
    return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.2
    out = np.sum(x * w) + b
    if out <= 0:
        return 0
    return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    out = AND(s1, s2)
    return out


if __name__ == '__main__':
    funcs = [
        AND,
        NAND,
        OR,
        XOR
    ]
    data = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1)
    ]

    for f in funcs:
        for d in data:
            print(f(d[0], d[1]))


