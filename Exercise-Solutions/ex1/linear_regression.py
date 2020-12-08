import numpy as np


def getDerivativeMat(x, y, theta):
    m = len(x)
    n = len(x[0])
    arr = [0] * n
    for i in range(m):
        for j in range(n):
            arr[j] += (np.dot(theta, x[i].transpose()) - y[i]) * x[i][j]

    return np.squeeze(np.asarray(arr))


def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)
    j_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * getDerivativeMat(x, y, theta)
        j_history[i] = computeCost(x, y, theta)

    return theta, j_history


def computeCost(x, y, theta):
    m = len(y)
    j = 0
    for i in range(m):
        j += (np.dot(theta, x[i].transpose()) - y[i]) ** 2
    j /= (2*m)
    return j


def featureNormalize(x):
    num_features = len(x[0])
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    mu_mat = np.matmul(np.ones((len(x), num_features)), np.diag(mu))
    sigma_mat = np.matmul(np.ones((len(x), num_features)), np.diag(sigma))

    x_norm = np.divide(x - mu_mat, sigma_mat)

    return x_norm, mu, sigma


def normaliseVal(x, mu, sigma):
    num_features = len(x[0])
    mu_mat = np.matmul(np.ones((len(x), num_features)), np.diag(mu))
    sigma_mat = np.matmul(np.ones((len(x), num_features)), np.diag(sigma))
    x_norm = np.divide(x - mu_mat, sigma_mat)
    return x_norm


def normalEqn(x, y):
    return np.matmul(
                np.linalg.inv(np.matmul(x.transpose(), x)),
                np.matmul(x.transpose(), y))
