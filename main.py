import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import NeuralNetwork as nn


start = datetime.now()
##This first bit is getting the training data to work with.
#image sizing
image_size = 28
number_of_labels = 10
number_of_pixels = 784

#base path
data_path = "data/"

#Load training data, uses this for loop because otherwise it will crash the shell
training_data = np.empty([60000,785])
row = 0
for line in open(data_path + "train.csv"):
    training_data[row] = np.fromstring(line, sep=",")
    row += 1

#Load testing data
test_data = np.empty([10000, 785])
row = 0
for line in open(data_path + "test.csv"):
    test_data[row] = np.fromstring(line, sep=",")
    row += 1
    
#This is used to adjust the MNIST dataset and map values into an interval between 0.01 and 1
adjust = 0.99 / 255

#Getting all the images in the correct interval
training_images = np.asfarray(training_data[:, 1:]) * adjust + 0.01
test_images = np.asfarray(test_data[:, 1:]) * adjust + 0.01

#Getting all the labels in the correct order
training_labels = np.asfarray(training_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

#Organising data in a one-hot representation. This will use 0.01s and 0.99 as this is better for calculatiosn

lr = np.arange(number_of_labels)

training_labels_one_hot = (lr==training_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

#making sure 0.01s and 0.99s are used

training_labels_one_hot[training_labels_one_hot==0] = 0.01
training_labels_one_hot[training_labels_one_hot==1] = 0.99

test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

timetaken = datetime.now() - start

print("Time taken:", timetaken)

#Here we will implement the nerual network code and use it to train.
neuralNetwork = nn.NeuralNetwork([784, 16, 16, 10], False, "data/weights.txt")
neuralNetwork.trainNetwork(training_images, training_labels_one_hot, 10, 1)
neuralNetwork = nn.NeuralNetwork([784, 16, 16, 10], True, "data/weights.txt")
neuralNetwork.testNetwork(test_images, test_labels)