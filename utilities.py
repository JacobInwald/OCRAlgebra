#This is a class that acts as a library for nescesary methods
import numpy as np
from numba import vectorize

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDerivative(x):
    return (x) * (1 - x)

def costFunctionOutputLayer(answer, trueValue):
    value = []
    for i in range(len(answer)):
        sign = (trueValue[i] - answer[i])
        sign = sign/abs(sign)
        value.append(sign * (trueValue[i] - answer[i])**2
    return value

def inputSummingFunction(inputNodes):
    total = 0
    for i in inputNodes:
        node = i[0]
        weight = i[1]
        total += node.output * weight
        
    return total
        
        
        
