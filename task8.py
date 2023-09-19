from task7 import dav_flet_paul

n = 2


def fun(x1, x2):
    return (10 + n) / (30 * n * ((x2 - x1 ** 2) ** 2 + n * (3 / 2 + x1) ** 2) + n)


def fun_to_maximize(x1, x2):
    return -1 * fun(x1, x2)


xMax = dav_flet_paul(fun_to_maximize, [[1.0, 0], [0, 1.0]], [-2.0, 1.0], 10e-6)
print(xMax, fun(*xMax))
