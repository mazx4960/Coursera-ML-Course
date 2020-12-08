from linear_regression import *
import numpy as np
import matplotlib.pyplot as plt
import unittest


########################################################################################################################
# Helper Functions
########################################################################################################################

def warmUpExercise():
    return np.identity(5, dtype=int)


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


def plotData(x, y):
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()


def plotBestFit(x, y, y_pred):
    plt.plot(x, y, 'rx')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.plot(x, y_pred, '-')
    plt.legend(['Training data', 'Linear Regression'])
    plt.show()


########################################################################################################################
# Test Code
########################################################################################################################

class TestEx1(unittest.TestCase):
    def setUp(self) -> None:
        self.x1, self.y1 = getDataFromFile('ex1data1.txt')
        self.x2, self.y2 = getDataFromFile('ex1data2.txt', dtype=int)

    def misc(self):
        # warm up
        # print(warmUpExercise())

        # 2.1 Plotting the data
        # x, y = getDataFromFile('ex1data1.txt')
        # plotData(x, y)
        pass

    def testCostFunction(self):
        # Part 2.2.3
        x = np.c_[np.ones(len(self.x1)), self.x1]

        theta1 = np.zeros(2)
        cost1 = computeCost(x, self.y1, theta1)
        self.assertAlmostEqual(32.07, cost1, places=2)

        theta2 = np.squeeze(np.asarray([-1, 2]))
        cost2 = computeCost(x, self.y1, theta2)
        self.assertAlmostEqual(54.24, cost2, places=2)

    def getGradientDescent(self):
        # Part 2.2.4 Gradient Descent
        x_ext = np.c_[np.ones(len(self.x1)), self.x1]
        theta = np.zeros(2)
        iterations = 1500
        alpha = 0.01

        final_theta, j_history = gradientDescent(x_ext, self.y1, theta, alpha, iterations)
        print('Theta computed from  gradient descent:', final_theta)

        # plotBestFit(self.x1, self.y1, np.matmul(x_ext, final_theta))

        # Predict values for population sizes of 35, 000 and 70, 000
        predict1 = np.dot(np.asarray([1, 3.5]), final_theta)
        print('For population = 35,000, we predict a profit of', predict1 * 10000)
        predict2 = np.dot(np.asarray([1, 7]), final_theta)
        print('For population = 70,000, we predict a profit of', predict2 * 10000)

    def getGradientDescentMulti(self):
        # Part 3.2
        x, mu, sigma = featureNormalize(self.x2)
        x_ext = np.c_[np.ones(len(x)), x]
        mu = np.insert(mu, 0, 0)
        sigma = np.insert(sigma, 0, 1)
        theta = np.zeros(3)
        iterations = 400
        alpha = 0.1

        final_theta, j_history = gradientDescent(x_ext, self.y2, theta, alpha, iterations)
        print('Theta computed from gradient descent:', final_theta)

        # Estimate the price of a 1650 sq - ft, 3 br house
        price = np.dot(normaliseVal(np.asarray([[1, 1650, 3]]), mu, sigma), final_theta)
        print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

        # plotData(range(iterations), j_history)

    @unittest.skip
    def testFeatureNormalize(self):
        featureNormalize(self.x2)

    def getNormalEquation(self):
        x_ext = np.c_[np.ones(len(self.x2)), self.x2]
        final_theta = normalEqn(x_ext, self.y2)
        print('Theta computed from gradient descent:', final_theta)

        # Estimate the price of a 1650 sq - ft, 3 br house
        price = np.dot(np.asarray([1, 1650, 3]), final_theta)
        print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

    def testGradAndNormEqn(self):
        x, mu, sigma = featureNormalize(self.x2)
        x_ext = np.c_[np.ones(len(x)), x]
        mu = np.insert(mu, 0, 0)
        sigma = np.insert(sigma, 0, 1)
        theta = np.zeros(3)
        iterations = 400
        alpha = 0.1

        final_theta, j_history = gradientDescent(x_ext, self.y2, theta, alpha, iterations)
        price_grad = float(np.dot(normaliseVal(np.asarray([[1, 1650, 3]]), mu, sigma), final_theta))

        x_ext = np.c_[np.ones(len(self.x2)), self.x2]
        final_theta = normalEqn(x_ext, self.y2)
        price_norm = float(np.dot(np.asarray([1, 1650, 3]), final_theta))

        self.assertAlmostEqual(price_grad, price_norm, places=1)


def main():
    test = TestEx1()
    test.setUp()
    test.getGradientDescentMulti()
    test.getNormalEquation()


if __name__ == '__main__':
    unittest.main()
    # main()
