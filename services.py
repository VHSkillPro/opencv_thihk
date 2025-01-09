import numpy as np

X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([[30, 32, 35, 37, 40, 43, 45, 47, 50, 53]]).T

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)


def grad(w):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


def cost(w):
    N = Xbar.shape[0]
    return 0.5 / N * np.linalg.norm(y - Xbar.dot(w), 2) ** 2
