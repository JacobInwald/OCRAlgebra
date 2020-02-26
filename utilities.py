# This is a class that acts as a library for nescesary methods
import numpy as np
from PIL import Image

# Neural network utilities


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

# Image manipulation


def loadImage(path):
    size = 28, 28
    im = Image.open(path).convert('LA')
    im.thumbnail(size)
    return im


def cleanImage(image):
    width, height = image.size
    array = []
    adjust = 0.99 / 255
    for y in range(height):
        for x in range(width):
            white, black = image.getpixel((x, y))
            if white == 255 and black == 255:
                array.append(0)
                continue
            elif white == 0 and black == 255:
                array.append(1)
                continue
            elif white > black:
                array.append((white - black) * adjust)
                continue
            else:
                array.append((black - white) * adjust)
                continue
    array = np.array(array)
    return array
