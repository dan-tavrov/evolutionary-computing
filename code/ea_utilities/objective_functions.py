import numpy as np


def simple_objective_function(arguments, n, norm_coef):
    return (arguments/norm_coef)**n


def F1(X):
    return np.sum(X**2, axis=1)


def F2(X):
    value = np.zeros(X.shape[0])

    for i in range(X.shape[1] - 1):
        value = value + \
            100*(X[:, i]**2 - X[:, i + 1])**2 + (1 - X[:, i])**2

    return value


def F3(X):
    return np.sum(np.floor(X), axis=1)


def F4(X):
    return np.sum(X**4 * (np.arange(X.shape[1]) + 1), axis=1) + np.random.randn(X.shape[0])


def F5(X):
    a = np.array([-32, -16, 0, 16, 32])
    A = np.vstack((np.tile(a, 5), np.repeat(a, 5)))

    result = 1/500 * np.ones(X.shape[0])

    for i in range(X.shape[0]):
        for j in range(25):
            result[i] += 1 / (j + 1 + (X[i, 0] - A[0, j])**6 + (X[i, 1] - A[1, j])**6)

    return 1 / result
