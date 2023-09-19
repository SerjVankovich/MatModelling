import numpy as np
from scipy import optimize

from task5 import golden_ratio


def func(x1: float, x2: float):
    return 100 * (x2 - x1 ** 2) ** 2 - (x1 - 1) ** 2


def test_func(x):
    x1 = x[0]
    x2 = x[1]
    return func(x1, x2)


def grad1(x1: float, x2: float):
    return -400 * x1 * x2 + 400 * x1 ** 3 - 2 * x1 + 2


def grad2(x1: float, x2: float):
    return 200 * x2 - 200 * x1 ** 2


def highest_grad_desc(fun, xStart: list, eps: float):
    x = np.array(xStart)
    step = eps + 10
    while step > eps:
        grads = np.array([grad1(x[0], x[1]), grad2(x[0], x[1])])
        grads = grads / np.linalg.norm(grads)

        def gradFun(s: float):
            return fun(x[0] - s * grads[0], x[1] - s * grads[1])

        step, _ = golden_ratio(gradFun, 0, 100, eps)

        x -= step * grads
    return x

# print(optimize.minimize(test_func, np.array([2, 2])))
# print(func(-1.241e+02, 1.541e+04))
# print(func(12.2, 149))

# xMin = highest_grad_desc(func, [-1000.0, 10000000.0], 10e-6)
# print(xMin, test_func(xMin))
