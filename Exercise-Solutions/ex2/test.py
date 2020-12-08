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
        self.x1, self.y1 = getDataFromFile('ex2data1.txt')
        self.x2, self.y2 = getDataFromFile('ex2data2.txt')

    def getVisualisation(self):
        # Part 1.1
        plotStudentData(self.x1, self.y1)


def main():
    test = TestEx2()
    test.setUp()
    test.getVisualisation()


if __name__ == '__main__':
    # unittest.main()
    main()