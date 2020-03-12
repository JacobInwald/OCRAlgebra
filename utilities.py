# This is a class that acts as a library for nescesary methods
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Neural network utilities


def sigmoid(x):
    # This is to stop overflows in this function
    return 1 / (1 + np.exp(-x))


def sigmoidInverse(x):
    return np.log(x / (1 - x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def outputCostDerivative(layer, trueValue):
    value = []
    # Loops through the outputs and calculates the derivative of the cost function for each of them
    for i in range(len(trueValue)):
        # This is the derivative of the cost function
        newCostDerivative = 2 * (trueValue[i] - layer[i].output) * sigmoidDerivative(layer[i].z)
        value.append(newCostDerivative)
    return value


def evaluateCost(answer, trueValue):
    value = 0
    for i in range(len(answer)):
        value += (answer[i] - trueValue[i]) ** 2
    return value

# Image manipulation


def loadImageFromPath(path):
    size = 28, 28
    im = Image.open(path).convert('LA')
    im.thumbnail(size)
    return im


def loadImageFromPIL(img):
    size = 28, 28
    img = img.convert('LA')
    img.thumbnail(size)
    return img

def centreImage(img):
    width, height = img.size
    left, right, top, bottom = 100000000000, 0, 0, 0
    tgtImg = Image.new('LA', (width, height))
    for y in range(height):
        for x in range(width):
            tgtImg.putpixel((x, y), (255, 255))
            white, black = img.getpixel((x, y))
            if white != 255 or black != 255:
                if x < left:
                    left = x
                if x > right:
                    right = x
                if top == 0:
                    top = y
                bottom = y
    img = img.crop((left, top, right, bottom))
    tgtImg.paste(img, (int((width / 2) - (right - left) / 2), int((height / 2) - (bottom - top) / 2)))
    return tgtImg


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
