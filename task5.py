import math
import matplotlib.pyplot as plt
import numpy as np


def func(x: float):
    return math.exp(math.sin(x)) + math.exp(-x)


def golden_ratio(fun, a: float, b: float, eps: float):
    phi: float = (1 + math.sqrt(5)) / 2
    resphi: float = 2 - phi

    x1: float = a + (b - a) * resphi
    x2: float = b - (b - a) * resphi
    y1: float = fun(x1)
    y2: float = fun(x2)

    i: int = 2

    while b - a > eps:
        i += 1
        if y1 < y2:
            b = x2
            x2 = x1
            y2 = y1
            x1 = a + (b - a) * resphi
            y1 = fun(x1)
        else:
            a = x1
            x1 = x2
            y1 = y2
            x2 = b - (b - a) * resphi
            y2 = fun(x2)
    return (x1 + x2) / 2, i


def print_func(fun, a, b):
    xs = np.linspace(a, b)
    ys = [fun(x) for x in xs]
    plt.plot(xs, ys)
    plt.show()


left: float = -math.pi / 2
right: float = 3 * math.pi

minX, i = golden_ratio(func, left, right, 10e-6)
minVal = func(minX)

plt.scatter(minX, minVal, color='red')
#print_func(func, left, right)
# plt.show()
