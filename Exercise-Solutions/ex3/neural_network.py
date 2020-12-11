import numpy as np
import scipy.optimize as op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lrCostFunction(theta, x, y, lam):
    """
    Compute cost and gradient for logistic regression with regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.

    :param theta: array_like
    :param x: array_like
    :param y: array_like
    :param lam: float
    :return: array_like
    """
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    # Computing the cost function j
    first_part = np.log(sigmoid(x.dot(theta)))
    second_part = np.log(1 - sigmoid(x.dot(theta)))
    first_part = first_part.reshape((m, 1))
    second_part = second_part.reshape((m, 1))
    std_j = -(np.sum(y * first_part + (1 - y) * second_part) / m)
    j = std_j + (lam / (2 * m)) * np.sum(theta[1:] ** 2)

    return j


def lrGrad(theta, x, y, lam):
    """
    Compute gradient for logistic regression with regularization
    J = LRGRAD(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.

    :param theta: array_like
    :param x: array_like
    :param y: array_like
    :param lam: float
    :return: array_like
    """
    m, n = np.shape(x)
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    # Computing gradient
    reg_term = (lam / m) * theta
    mask = np.diag(np.append([0], np.ones((n-1))))
    grad = ((x.transpose()).dot(sigmoid(x.dot(theta)) - y) / m) + np.matmul(mask, reg_term)

    return grad.flatten()


def oneVsAll(x, y, num_labels, lambd):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i
    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds
    to the classifier for label i

    :param x: array_like
    :param y: array_like
    :param num_labels: int
    :param lambd: float
    :return: array_like
    """
    m, n = np.shape(x)
    all_theta = np.zeros((num_labels, n + 1))
    x_ext = np.c_[np.ones((m, 1)), x]

    for i in range(num_labels):
        initial_theta = np.zeros((n + 1, 1))
        theta = op.fmin_cg(lrCostFunction, fprime=lrGrad, x0=initial_theta, args=(x_ext, y == (i+1), lambd), maxiter=50)
        all_theta[i] = theta

    return all_theta


def predictOneVsAll(all_theta, x):
    """
    PREDICT Predict the label for a trained one-vs-all classifier. The labels
    are in the range 1..K, where K = size(all_theta, 1).
    p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class. You should set p to a vector
    of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    for 4 examples)

    :param all_theta: array_like
    :param x: array_like
    :return: array_like
    """
    m, n = np.shape(x)
    x_ext = np.c_[np.ones((m, 1)), x]
    result = sigmoid(x_ext.dot(all_theta.T)).argmax(axis=1) + np.ones(m)
    return result.reshape((m, 1))


def predict(theta1, theta2, x):
    """
    PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)

    Neural network architecture: 3 layers- an input layer, a hidden layer and an output layer.
    Inputs are pixel values of digit images. Since the images are of size 20 x 20,
    this gives us 400 input layer units (excluding the extra bias unit).

    The parameters have dimensions that are sized for a neural network with 25 units in the second layer and
    10 output units (corresponding to the 10 digit classes).

    :param theta1: array_like (25 x 401)
    :param theta2: array_like (10 x 26)
    :param x: array_like
    :return: array_like
    """
    m, n = np.shape(x)
    x_ext = np.c_[np.ones((m, 1)), x]
    activation = sigmoid(np.c_[np.ones((m, 1)), (x_ext.dot(theta1.T))])
    out = (sigmoid(activation.dot(theta2.T))).argmax(axis=1) + np.ones(m)  # account for the off by one indexing
    return out.reshape((m, 1))
