from logistic_regression import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import unittest


########################################################################################################################
# Helper Functions
########################################################################################################################

def getDataFromFile(filename, dtype=float):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            x_val, y_val = list(map(dtype, data[:-1])), int(data[-1])
            x.append(x_val)
            y.append(y_val)
    return np.squeeze(np.asarray(x)), np.squeeze(np.asarray(y))


def plotStudentData(x, y):
    admitted = x[y[:] == 1, :]
    not_admitted = x[y[:] == 0, :]

    plt.plot(admitted[:, 0], admitted[:, 1], 'k+')
    plt.plot(not_admitted[:, 0], not_admitted[:, 1], 'ko')

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()


def plotDecisionBoundary(x, y, final_theta):
    xx = np.linspace(np.min(x), np.max(x))
    grad = -final_theta[1] / final_theta[2]
    yy = grad * xx - final_theta[0] / final_theta[2]

    plt.plot(xx, yy, 'k-')
    plotStudentData(x, y)


def plotMicrochipsData(x, y):
    accepted = x[y[:] == 1, :]
    rejected = x[y[:] == 0, :]

    plt.plot(accepted[:, 0], accepted[:, 1], 'k+')
    plt.plot(rejected[:, 0], rejected[:, 1], 'ko')

    plt.xlabel('Microchips Test 1')
    plt.ylabel('Microchips Test 2')
    plt.legend(['Accepted', 'Rejected'])
    plt.show()


def plotBoundary2(theta, x, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (mapFeature(np.array([u[i]]), np.array([v[j]])).dot(theta))

    z = z.T
    plt.contour(u, v, z)
    plotMicrochipsData(x, y)


########################################################################################################################
# Test Code
########################################################################################################################

class TestEx2(unittest.TestCase):
    def setUp(self) -> None:
        self.x1, self.y1 = getDataFromFile('ex2data1.txt')
        self.x2, self.y2 = getDataFromFile('ex2data2.txt')

    def getVisualisation(self):
        # Part 1.1
        plotStudentData(self.x1, self.y1)

    def testSigmoid(self):
        # Part 1.2.1 Sigmoid function
        self.assertAlmostEqual(0.5, sigmoid(0))

        sigmoid_mat = sigmoid(np.asarray([-10000, 0, 10000]))
        np.testing.assert_array_almost_equal(np.asarray([0, 0.5, 1]), sigmoid_mat)

    def testCostFunction(self):
        # Part 1.2.2 Cost function and gradient
        m, n = np.shape(self.x1)
        x = np.c_[np.ones(m), self.x1]
        initial_theta = np.zeros(n + 1)

        cost = costFunction(initial_theta, x, self.y1)
        grad = getGrad(initial_theta, x, self.y1)
        self.assertAlmostEqual(0.693, cost, places=3)
        np.testing.assert_array_almost_equal(
            np.asarray([-0.1000, -12.0092, -11.2628]),
            grad, decimal=4)

    def testfmin_tnc(self):
        m, n = np.shape(self.x1)
        x = np.c_[np.ones(m), self.x1]
        initial_theta = np.zeros((n + 1))
        final_theta, cost = fmin_tnc(initial_theta, costfun=costFunction, gradfun=getGrad, args=(x, self.y1))

        # print('Cost at theta found by fmin_tnc:', cost)
        # print('Theta computed from truncated newton algorithm:', final_theta)
        self.assertAlmostEqual(0.203, cost, places=3)

    def getDecisionBoundary(self):
        m, n = np.shape(self.x1)
        x = np.c_[np.ones(m), self.x1]
        initial_theta = np.zeros((n + 1))
        final_theta, cost = fmin_tnc(initial_theta, costfun=costFunction, gradfun=getGrad, args=(x, self.y1))

        plotDecisionBoundary(self.x1, self.y1, final_theta)

    def testAccuracy(self):
        m, n = np.shape(self.x1)
        x = np.c_[np.ones(m), self.x1]
        initial_theta = np.zeros((n + 1))
        final_theta, cost = fmin_tnc(initial_theta, costfun=costFunction, gradfun=getGrad, args=(x, self.y1))

        # Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
        prob = sigmoid(np.matmul(np.asarray([1, 45, 85]), final_theta))
        # print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
        self.assertAlmostEqual(0.776, prob, places=3)

        # Compute accuracy on our training set
        p = predict(final_theta, x)
        accuracy = np.mean(np.equal(p, self.y1).astype(int)) * 100
        print('Train Accuracy: ', accuracy)

    def getVisualization2(self):
        plotMicrochipsData(self.x2, self.y2)

    def testMapFeature(self):
        x1 = self.x2[:, 0]
        x2 = self.x2[:, 1]
        n = np.shape(x1)[0]
        mapped = mapFeature(x1, x2)
        self.assertEqual((n, 28), np.shape(mapped))

    def testCostFunctionReg(self):
        x1 = self.x2[:, 0]
        x2 = self.x2[:, 1]
        mapped = mapFeature(x1, x2)
        m, n = np.shape(mapped)

        lam = 1
        initial_theta = np.zeros((n, 1))
        cost = costFunctionReg(initial_theta, mapped, self.y2, lam)
        grad = getGradReg(initial_theta, mapped, self.y2, lam)

        self.assertAlmostEqual(0.693, cost, places=3)
        self.assertEqual((28,), np.shape(grad))

    def testAccuracy2(self):
        x1 = self.x2[:, 0]
        x2 = self.x2[:, 1]
        mapped = mapFeature(x1, x2)
        m, n = np.shape(mapped)

        lam = 1
        initial_theta = np.zeros(n)
        final_theta, cost = fmin_tnc(initial_theta, costfun=costFunctionReg, gradfun=getGradReg,
                                     args=(mapped, self.y2, lam))
        # print(final_theta, cost)

    def getDecisionBoundary2(self):
        x1 = self.x2[:, 0]
        x2 = self.x2[:, 1]
        mapped = mapFeature(x1, x2)
        m, n = np.shape(mapped)

        lam = 1
        initial_theta = np.zeros(n)
        final_theta, cost = fmin_tnc(initial_theta, costfun=costFunctionReg, gradfun=getGradReg,
                                     args=(mapped, self.y2, lam))
        plotBoundary2(final_theta, self.x2, self.y2)


def main():
    test = TestEx2()
    test.setUp()
    test.getDecisionBoundary2()


if __name__ == '__main__':
    unittest.main()
    # main()
