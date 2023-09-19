import numpy as np

from task5 import golden_ratio
from task6 import grad1, grad2, func


def dav_flet_paul(fun, H, xStart, eps):
    n = 5
    grads = [1, 1]
    x = np.array(xStart)
    npH = np.array(H)
    num_iters = 0

    while np.linalg.norm(grads) > eps:
        if num_iters > 1000:
            return x
        grads = np.array([grad1(x[0], x[1]), grad2(x[0], x[1])])

        y = x
        for i in range(n):

            S = np.dot(-1 * npH, grads.T).T

            def gradFun(s: float):
                return fun(x[0] + s * S[0], x[1] + s * S[1])

            l, _ = golden_ratio(gradFun, 0, 100, eps)

            y += l * S

            P = l * S

            q = np.array([grad1(y[0], y[1]), grad2(y[0], y[1])]) - grads

            dH = P.T.dot(P) / P.dot(q) - np.vstack(npH.dot(q)).dot(np.dstack(q)).dot(npH) / q.dot(npH).dot(q.T)

            npH += dH.reshape((2, 2))
        x = y
        num_iters = num_iters + 1
    return x


#xMin = dav_flet_paul(func, [[1.0, 0], [0, 1.0]], [2.0, 3.0], 10e-6)
#print(xMin, func(xMin[0], xMin[1]))
