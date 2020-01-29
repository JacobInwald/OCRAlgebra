import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import NeuralNetwork as nn


start = datetime.now()
# This first bit is getting the training data to work with.
# image sizing
imageSize = 28
numberOfLabels = 10
numberOfPixels = 784

# base path
dataPath = "data/"

# Load training data, uses this for loop because otherwise it will crash the shell
trainingData = np.empty([60000, 785])
row = 0
for line in open(dataPath + "train.csv"):
    trainingData[row] = np.fromstring(line, sep=",")
    row += 1

# Load testing data
testData = np.empty([10000, 785])
row = 0
for line in open(dataPath + "test.csv"):
    testData[row] = np.fromstring(line, sep=",")
    row += 1
    
# This is used to adjust the MNIST dataset and map values into an interval between 0.01 and 1
adjust = 0.99 / 255

# Getting all the images in the correct interval
trainingImages = np.asfarray(trainingData[:, 1:]) * adjust + 0.01
testImages = np.asfarray(testData[:, 1:]) * adjust + 0.01

# Getting all the labels in the correct order
trainingLabels = np.asfarray(trainingData[:, :1])
testLabels = np.asfarray(testData[:, :1])

# Organising data in a one-hot representation. This will use 0.01s and 0.99 as this is better for calculatiosn

lr = np.arange(numberOfLabels)

trainingLabelsOneHot = (lr == trainingLabels).astype(np.float)
testLabelsOneHot = (lr == testLabels).astype(np.float)

# making sure 0.01s and 0.99s are used

trainingLabelsOneHot[trainingLabelsOneHot == 0] = 0
trainingLabelsOneHot[trainingLabelsOneHot == 1] = 0.99

testLabelsOneHot[testLabelsOneHot == 0] = 0
testLabelsOneHot[testLabelsOneHot == 1] = 0.99

timeTaken = datetime.now() - start

print("Time taken:", timeTaken)

# Here we will implement the neural network code and use it to train.
neuralNetwork = nn.NeuralNetwork([784, 20, 16, 10], True, "data/weights.txt")
neuralNetwork.trainNetwork(trainingImages, trainingLabelsOneHot, 20, 0.001)
neuralNetwork = nn.NeuralNetwork([784, 20, 16, 10], True, "data/weights.txt")
neuralNetwork.testNetwork(testImages, testLabels)
