# This is a class that acts as a library for nescesary methods
import numpy as np
from numba import vectorize


def sigmoid(x):
    # This is to stop overflows in this function
    if x >= 1000:
        return 1
    elif x <= -1000:
        return 0
    return 1 / (1 + np.exp(-x))


def sigmoidInverse(x):
    if x >= 0.9999999999999999:
        return 37
    return np.log(x / (1 - x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def outputCostDerivative(answer, trueValue, zList):
    value = []
    # Loops through the outputs and calculates the derivative of the cost function for each of them
    for i in range(len(answer)):
        # This is intended output
        y = trueValue[i]
        # This is the activation of the node
        a = answer[i]
        # This is the summed value of the weights times the inputs added to the bias of the node
        z = zList[i]
        # This is the derivative of the cost function
        newCostDerivative = 2 * (y - a) * sigmoidDerivative(z)
        value.append(newCostDerivative)
    return value


def evaluateCost(answer, trueValue):
    value = 0
    for i in range(len(answer)):
        value += (answer[i] - trueValue[i]) ** 2
    return value
