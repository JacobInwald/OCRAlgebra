#This is a class that acts as a library for nescesary methods
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inputSummingFunction(inputNodes):
    total = 0
    for i in inputNodes:
        node = i[0]
        weight = i[1]
        if(total > 100000000000000000000000000000):
            total = 100000000000000000000000000000
        if(total < -100000000000000000000000000000):
            total = -100000000000000000000000000000
        if(weight > 100000000000000000000000000000):
            weight = 100000000000000000000000000000
        if(weight < -1000000000000000000000000000000):
            weight = -100000000000000000000000000000
            
        total += node.output * weight
        
    return total
        
        
        
