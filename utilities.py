# This is a class that acts as a library for nescesary methods
import numpy as np
from numba import vectorize


def sigmoid(x):
    if x >= 100000:
        return 1
    elif x <= -100000:
        return 0
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def outputCostDerivative(answer, trueValue):
    value = []
    for i in range(len(answer)):
        y = trueValue[i]
        a = answer[i]
        z = sigmoidDerivative(a)
        newCost = 2 * (y - a) * sigmoidDerivative(z)
        value.append(newCost)
    return value


def evaluateCost(answer, trueValue):
    value = 0
    for i in range(len(answer)):
        value += (answer[i] - trueValue[i]) ** 2
    return value
