import numpy as np
import scipy.optimize as op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def getDerivativeMat(x, y, theta):
    m, n = np.shape(x)
    arr = [0] * n
    for i in range(m):
        for j in range(n):
            arr[j] += (sigmoid(np.dot(theta, x[i])) - y[i]) * x[i][j]

    return np.squeeze(np.asarray(arr))


def costFunction(theta, x, y):

    m = len(y)
    j = 0
    theta = np.zeros(np.size(theta))

    for i in range(m):
        j += (-y[i] * np.log(sigmoid(np.dot(theta, x[i])))
              - (1 - y[i]) * np.log(1 - sigmoid(np.dot(theta, x[i]))))
    j /= m

    grad = getDerivativeMat(x, y, theta) / m
    # grad = np.dot(x.transpose(), (sigmoid(np.dot(x, theta)) - y)) / m

    return j, grad


def fmin(initial_theta, x, y):
    result = op.fmin_tnc(func=costFunction,
                         x0=initial_theta,
                         args=(x, y))
    return result
