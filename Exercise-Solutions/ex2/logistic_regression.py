import numpy as np
import scipy.optimize as op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def getGrad(theta, x, y):
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    grad = (x.transpose()).dot(sigmoid(x.dot(theta)) - y) / m
    return grad.flatten()


def costFunction(theta, x, y):
    """
    Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.

    :param theta:
    :param x:
    :param y:
    :return:
    """
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    first_part = np.log(sigmoid(x.dot(theta)))
    second_part = np.log(1 - sigmoid(x.dot(theta)))
    first_part = first_part.reshape((m, 1))
    second_part = second_part.reshape((m, 1))
    j = -(np.sum(y * first_part + (1 - y) * second_part) / m)

    return j


def fmin_tnc(initial_theta, costfun, gradfun, args):
    result = op.minimize(fun=costfun, x0=initial_theta, args=args, method='TNC', jac=gradfun,
                         options={'maxiter': 400})
    final_theta = result.x
    cost = result.fun
    return final_theta, cost


def predict(theta, x):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta
    p = PREDICT(theta, X) computes the predictions for X using a
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    :param theta:
    :param x:
    :return:
    """
    probs = sigmoid(x.dot(theta))
    return (probs > 0.5).astype(int)


def mapFeature(x1, x2):
    """
    MAPFEATURE(X1, X2) maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size

    :param x1: vector of n dimension
    :param x2: vector of n dimension
    :return: matrix of (n * 28)
    """
    degree = 6
    n = np.shape(x1)[0]
    out = np.ones((n, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            new = (x1 ** (i - j)) * (x2 ** j)
            new = new.reshape((n, 1))
            out = np.append(out, new, axis=1)

    return out


def costFunctionReg(theta, x, y, lam):
    """
    Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.

    :param theta:
    :param x:
    :param y:
    :param lam:
    :return:
    """
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))

    j = costFunction(theta, x, y) + (lam / (2 * m)) * np.sum(theta[1:] ** 2)

    return j


def getGradReg(theta, x, y, lam):
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))
    reg_term = (lam / m) * theta
    mask = np.diag(np.append([0], np.ones((n - 1))))
    grad = getGrad(theta, x, y).reshape((n, 1)) + np.matmul(mask, reg_term)
    return grad.flatten()

