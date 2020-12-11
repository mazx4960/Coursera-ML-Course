from neural_network import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import unittest
import warnings


########################################################################################################################
# Helper Functions
########################################################################################################################

def get_mat_from_file(filename):
    data = sio.loadmat(filename)
    keys = [key for key in data.keys() if key[:2] != '__']
    return [data[key] for key in keys]


########################################################################################################################
# Test Code
########################################################################################################################

class TestEx3(unittest.TestCase):
    def setUp(self) -> None:
        self.x1, self.y1 = get_mat_from_file('ex3data1.mat')
        self.theta1, self.theta2 = get_mat_from_file('ex3weights.mat')

    def testlrCostFunction(self):
        theta_t = np.array([-2, -1, 1, 2])
        x_t = np.c_[np.ones((5, 1)), np.reshape(range(1, 16), (3, 5)).T / 10]
        y_t = np.asarray([1, 0, 1, 0, 1]).reshape((5, 1))
        lambda_t = 3
        j = lrCostFunction(theta_t, x_t, y_t, lambda_t)
        grad = lrGrad(theta_t, x_t, y_t, lambda_t)

        self.assertAlmostEqual(2.534819, j, places=6)
        np.testing.assert_array_almost_equal([0.146561, -0.548558, 0.724722, 1.398003], grad, decimal=6)

    def testOneVsAll(self):
        num_labels = 10
        lambd = 0.1
        all_theta = oneVsAll(self.x1, self.y1, num_labels, lambd)
        pred = predictOneVsAll(all_theta, self.x1)
        print('Training Set Accuracy:', np.mean(np.equal(pred, self.y1).astype(int)) * 100)

    def testFeedForward(self):
        pred = predict(self.theta1, self.theta2, self.x1)
        print('Training Set Accuracy:', np.mean(np.equal(pred, self.y1).astype(int)) * 100)


def main():
    test = TestEx3()
    test.setUp()
    test.testFeedForward()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    unittest.main()
    # main()
