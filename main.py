import numpy as np
from datetime import datetime
import NeuralNetwork as nn
import gui


start = datetime.now()
# This first bit is getting the training data to work with.
# image sizing
imageSize = 28
numberOfLabels = 10
numberOfPixels = 784

# base path
dataPath = "data/"
#
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
# Organising data in a one-hot representation. This will use 0.01s and 0.99 as this is better for calculation
lr = np.arange(numberOfLabels)

trainingLabelsOneHot = (lr == trainingLabels).astype(np.float)
testLabelsOneHot = (lr == testLabels).astype(np.float)
# making sure 0.01s and 0.99s are used
# trainingLabelsOneHot[trainingLabelsOneHot == 0] = 0.01
# trainingLabelsOneHot[trainingLabelsOneHot == 1] = 0.99
# testLabelsOneHot[testLabelsOneHot == 0] = 0.01
# testLabelsOneHot[testLabelsOneHot == 1] = 0.99

timeTaken = datetime.now() - start
print("Time taken:", timeTaken)
# Here we will implement the neural network code and use it to train.
# The second value determines whether the network will be loaded from a file. To make it do a proper training session
# change it to False
neuralNetwork = nn.NeuralNetwork([784, 24, 24, 10], True, "data/weights[1].txt")
# Loss: 0.016759857738949845 for most trained one
# To edit the learning paces change the values in the learningPace.txt
neuralNetwork.trainNetwork(trainingImages, trainingLabelsOneHot, 100, 0.1, 0.016759857738949845)
neuralNetwork.testNetwork(testImages, testLabelsOneHot, 1)
gui.start(neuralNetwork)
