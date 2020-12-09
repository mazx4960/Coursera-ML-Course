from logistic_regression import *
import numpy as np
import matplotlib.pyplot as plt
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
            x_val, y_val = list(map(dtype, data[:-1])), dtype(data[-1])
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


########################################################################################################################
# Test Code
########################################################################################################################

class TestEx2(unittest.TestCase):
    def setUp(self) -> None:
        self.x1, self.y1 = getDataFromFile('ex2data1.txt', dtype=np.float128)
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

        cost, grad = costFunction(initial_theta, x, self.y1)
        self.assertAlmostEqual(0.693, cost, places=3)
        np.testing.assert_array_almost_equal(
            np.asarray([-0.1000,  -12.0092, -11.2628]),
            grad, decimal=4)

    def testfmin(self):
        m, n = np.shape(self.x1)
        x = np.c_[np.ones(m), self.x1]
        initial_theta = np.zeros((n + 1), dtype=int)
        print(fmin(initial_theta, x, self.y1))


def main():
    test = TestEx2()
    test.setUp()
    test.getVisualisation()


if __name__ == '__main__':
    unittest.main()
    # main()
