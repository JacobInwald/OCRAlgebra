import matplotlib.pyplot as plt
import utilities as util
import random
from tqdm import tqdm
import numpy as np


class Node:

    def __init__(self, inputNodes, outputNodes, output, bias, number):
        # These will be a lists of tuples, the first item being the node the second being the weight
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        # These are the variables used in propagation. I am using the sigmoid function as an activation function.
        self.bias = bias
        self.activationFunction = util.sigmoid
        self.output = output
        self.z = util.sigmoidDerivative(output)
        self.error = 0
        # Momentum variable used to make sure the node continues on its trend from preivous iterations.
        self.momentum = 0
        self.prevOutput = 0
        # This is an id that stores position in the nodes array
        self.number = number

    def feedForward(self):
        # This just sums the input nodes outputs and then passes that values through an activation function
        self.prevOutput = self.output
        self.z = self.sumInputs() + self.bias
        self.output = self.activationFunction(self.z)

    def sumInputs(self):
        # Loops through all the inputs and sums the weights and activation
        total = 0
        for i in self.inputNodes:
            total += i[1] * i[0].output
        return total


class NeuralNetwork:

    def __init__(self, nodeNumbers, loadFromFile, filePath):
        # This uses the node numbers as an input to generate a network with randomised weights and biases
        self.nodes = [[] for _ in nodeNumbers]
        self.layers = len(nodeNumbers)
        self.generateNetwork(nodeNumbers)
        # This uses the next two parameters to check whether the network is being loaded from a file and if
        # not overwrites to the file specified
        self.path = filePath
        if loadFromFile:
           self.loadFromFile(self.path)
        # This initialises an empty array that can be used to plot data with matplot
        self.costArray = []

    # generates a connected network based on an array of numbers e.g. [784, 16, 16, 10]
    def generateNetwork(self, nodeNumbers):
        # generates an array of nodes that is of the same size specified
        index = -1
        for i in nodeNumbers:
            index += 1
            for x in range(i):
                # initialise nodes with None type inputs and outputs as well as biases and outputs
                self.nodes[index].append(Node([None], [None], random.uniform(0, 1), random.uniform(-10, 10), x))

        # This loops through the generated array and sets the input nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the first layer which are the inputs and so don't have input nodes
            if i == 0:
                continue
            # sets inputNodes
            for tgtNode in layer:
                # initialise inputNodes as a None array with length of previous layer
                tgtNode.inputNodes = [None] * len(self.nodes[i - 1])
                # loops through previous layer and sets weight and nodes
                for x in range(len(self.nodes[i - 1])):
                    tgtNode.inputNodes[x] = [self.nodes[i - 1][x], random.uniform(-0.5, 0.5)]

        # This loops through the generated list and sets the output nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the last layer which are the output nodes
            if i == self.layers - 1:
                break
            # set outputNodes
            for tgtNode in layer:
                # initialise outputNodes as a None array with length of previous layer
                tgtNode.outputNodes = [None] * len(self.nodes[i + 1])
                # loops through next layer and sets weights and nodes
                for x in range(len(self.nodes[i + 1])):
                    tgtNode.outputNodes[x] = [self.nodes[i + 1][x], self.nodes[i + 1][x].inputNodes[tgtNode.number][1]]

    def saveToFile(self, filePath):
        # This initialises a string that we will write to the file
        stringToWrite = ""
        # Loops through the nodes array to add the inputNodes weights to the string
        for i in range(self.layers):
            layer = self.nodes[i]
            # Skips the first layer which has no inputs
            if i == 0:
                continue
            # I am using the # as a separator in between layers
            stringToWrite += "#"
            # Loops through the layer to get all the nodes
            for node in layer:
                # I am using the ~ as a separator in between nodes
                stringToWrite += "~"
                # Loops through inputNodes to get all the weights
                for data in node.inputNodes:
                    # I am using the , as a separator in between weights
                    stringToWrite += ","
                    stringToWrite += str(data[1])

        # This is the separator between biases and weights
        stringToWrite += "|"
        # Loops through the nodes array to get the layers
        for layers in self.nodes:
            # I am using the # as a separator between layers
            stringToWrite += "#"
            # Loops through the layers to get the nodes
            for node in layers:
                # I am using the ~ as a separator between nodes/biases
                stringToWrite += "~"
                # Appends the bias to the string
                stringToWrite += str(node.bias)

        # Opens the file for overwriting
        file = open(filePath, 'w')
        file.write(stringToWrite)
        file.close()

    def loadFromFile(self, filePath):
        # This opens the file and reads the entire thing as one string
        file = open(filePath, 'r')
        weights = file.read()
        file.close()
        # Splits for in between biases and weights
        weights = weights.split("|")
        # This entire thing loops through the new data and splits it for each different data type
        # The while loops remove the extraneous data that comes from splitting strings
        for i in range(len(weights)):
            # This is specifically for weights
            if i == 0:
                # This splits between layers
                weights[i] = weights[i].split("#")
                for x in range(len(weights[i])):
                    # This splits between nodes
                    weights[i][x] = weights[i][x].split("~")
                    for y in range(len(weights[i][x])):
                        # This splits between weights
                        weights[i][x][y] = weights[i][x][y].split(",")
                        while '' in weights[i][x][y]:
                            weights[i][x][y].remove('')
                    while '' in weights[i][x]:
                        weights[i][x].remove('')
                while '' in weights[i]:
                    weights[i].remove('')
            # This is specifically for biases
            if i == 1:
                # This splits between layers
                weights[i] = weights[i].split("#")
                for x in range(len(weights[i])):
                    # This splits between nodes/biases
                    weights[i][x] = weights[i][x].split("~")
                    while '' in weights[i][x]:
                        weights[i][x].remove('')
                while '' in weights[i]:
                    weights[i].remove('')

        # This is the final method of removing extraneous data
        weights = [x for x in weights if x]

        # With the data taken and pruned I can now add it to the network.
        for i in range(len(weights)):
            # This is for the weights
            if i == 0:
                # Loops through the layers to set the inputNodes
                for x in range(len(weights[i])):
                    # I want to skip the first layer as it has no input Nodes
                    if x == 0:
                        continue
                    # More data pruning
                    weights[i][x] = [a for a in weights[i][x] if a]
                    for y in range(len(weights[i][x])):
                        for z in range(len(weights[i][x][y])):
                            # Sets the inputs Nodes at the same time as converting them to floats
                            self.nodes[x][y].inputNodes[z][1] = float(weights[i][x][y][z])
            if i == 1:
                # More data pruning
                weights[i] = [a for a in weights[i] if a]
                for x in range(len(weights[i])):
                    # More data pruning
                    weights[i][x] = [a for a in weights[i][x] if a]
                    for y in range(len(weights[i][x])):
                        # Sets the input Nodes at the same time as converting them to floats
                        self.nodes[x][y].bias = float(weights[i][x][y])

        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the last layer which are the output nodes
            if i == self.layers - 1:
                break
            # set outputNodes
            for tgtNode in layer:
                for x in range(len(self.nodes[i + 1])):
                    # loops through next layer and sets weights and nodes.
                    tgtNode.outputNodes[x][1] = self.nodes[i + 1][x].inputNodes[tgtNode.number][1]

    def loadInputs(self, inputArray):
        # Loads an input array into the neural network
        for i in range(len(inputArray)):
            self.nodes[0][i].output = inputArray[i]

    def getNewPaces(self, path):
        data = open(path)
        data = data.readlines()
        lp = float(data[0])
        mp = float(data[1])
        return lp, mp

    def feedForward(self):
        output = []
        # does all the feed forwards for all the nodes
        for layer in self.nodes:
            # skips first layer which already has their own outputs
            if self.nodes.index(layer) == 0:
                continue

            for node in layer:
                node.feedForward()

    def backPropagateCost(self, trueValue):
        # Get the output layers error values for usage in the back propagation
        # Loops through the outputs and calculates the derivative of the cost function for each of them
        for i in range(len(trueValue)):
            # This is the derivative of the cost function
            newCostDerivative = 2 * (trueValue[i] - self.nodes[self.layers - 1][i].output) * util.sigmoidDerivative(self.nodes[self.layers - 1][i].z)
            self.nodes[self.layers - 1][i].error = newCostDerivative

        # loops through the nodes array to propagate backwards for the error of each node
        for i in range(self.layers - 1, -1, -1):
            # Skips the last layer as it already has its error
            if i == self.layers - 1 or i == 0:
                continue
            # loops through the nodes to set error
            for tgtNode in self.nodes[i]:
                cost = 0
                # Sums the error of the node
                for x in tgtNode.outputNodes:
                    # This multiplies previous nodes error with the weight connecting both of the nodes to get the error
                    # of the node, this is because of the chain rule.
                    cost += x[0].error * x[1]
                # tgtNode.momentum = tgtNode.error
                tgtNode.error += cost * util.sigmoidDerivative(tgtNode.z)

    def updateWeights(self, learningPace, momentumPace, batchSize):
        # This loops through the generated list and sets the input nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the first layer which is just inputs and are not actually nodes, so don't have input nodes.
            if i == 0:
                continue
            # updates weights
            for tgtNode in layer:
                # loops through previous layer and sets weight and nodes
                for x in tgtNode.inputNodes:
                    # multiplies the nodes error with the connected nodes output to get the weights specific error
                    # This is temporary code to check out momentum
                    x[1] += learningPace * (tgtNode.error / batchSize) * x[0].output + learningPace * momentumPace * tgtNode.momentum * x[0].prevOutput
                    x[0].outputNodes[tgtNode.number][1] = x[1]
                # I already have the bias error so I just multiply it by a constant to get change
                tgtNode.bias += learningPace * (tgtNode.error / batchSize)


    def trainNetwork(self, trainingData, trainingLabels, epochs, learningPace, batchSize, momentumPace, lowestCost):
        guess = []
        costArray = []
        costMean = 0
        lp = learningPace
        mp = momentumPace
        lowestCost = lowestCost
        for x in tqdm(range(epochs)):
            costMean = 0
            for i in range(len(trainingData)):
                # gets guess and true values
                # loads the right input array for the feed forward algorithm
                self.loadInputs(trainingData[i])
                # generates a guess using the feed forward algorithm
                self.feedForward()
                # gets the right answers from the array
                trueValue = trainingLabels[i]
                # back propagates error to get the error of each node
                self.backPropagateCost(trueValue)
                # updates the weights and biases using the error of the nodes
                if (i + 1) % batchSize == 0:
                    self.updateWeights(lp, mp, batchSize)
                cost = util.evaluateCost(guess, trueValue)
                costMean += cost

                if (i + 1) % 10000 == 0 or i == 0:
                    if i != 0:
                        costMean = costMean / 10000
                    paces = self.getNewPaces("data/learningPace.txt")
                    lp = paces[0]
                    mp = paces[1]
                    costArray.append(costMean)
                    print("Epoch:", x)
                    print("Rep:", i + 1)
                    print("Guess:", self.getAnswer(trainingData[i]))
                    print("Answer:", trueValue)
                    print("Learning Pace:", lp)
                    print("Loss:", costMean)
                    if costMean <= lowestCost and i != 0:
                        lowestCost = costMean
                        print("Saving weights ...")
                        self.saveToFile(self.path)
                    if costMean <= 0.001 and i != 0:
                        return
                    costMean = 0
                    print("-----------------------------------")

        plt.plot(costArray)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.show()

    def testNetwork(self, testData, testLabels, rightNumber):
        right = 0
        numbersRight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # This is for testing percentages of the neural network getting it correctly
        for i in range(len(testData)):
            self.loadInputs(testData[i])
            guess = self.feedForward()
            correct = testLabels[i].index(rightNumber)
            runningTotal = 0
            for x in guess:
                if x >= runningTotal:
                    runningTotal = x
                    answer = guess.index(x)
            print("Guess", answer, guess)
            print("Answer", correct)

            if correct == answer:
                right += 1
                numbersRight[correct - 1] += 1
            if i >= 9980:
                img = testData[i].reshape((28, 28))
                plt.imshow(img, cmap="Greys")
                print("Right Answer: ", correct)
                print("Guess: ", answer)
                plt.show()
                print("----------------")
        print("Total percentage correct:", (right * 100) / len(testData), "%")
        for correct in numbersRight:
            print("Percentage correct for", numbersRight.index(correct) + 1, "is:", (correct * 100) / (len(testData) / 10), "%")
        plt.show()
        input("Enter to close the program")

    def getAnswer(self, input):
        self.loadInputs(input)
        output = []
        # does all the feed forwards for all the nodes
        for layer in self.nodes:
            # skips first layer which already has their own outputs
            if self.nodes.index(layer) == 0:
                continue

            for node in layer:
                node.feedForward()
        # gets the output for all the output nodes
        for i in self.nodes[len(self.nodes) - 1]:
            output.append(i.output)
        x = 0
        for i in output:
            if i > x:
                x = i
        guess = output.index(x)
        return str(guess) + " with " + str(x) + " activation."



dataset = [[1.38807019, 1.850220317],
           [3.06407232, 3.005305973],
           [7.627531214, 2.759262235],
           [5.332441248, 2.088626775],
           [6.922596716, 1.77106367],
           [3.396561688, 4.400293529],
           [8.675418651, -0.242068655],
           [2.7810836, 2.550537003],
           [7.673756466, 3.508563011],
           [1.465489372, 2.362125076]]
trueValue = [[0.01, 0.99],
             [0.01, 0.99],
             [0.01, 0.99],
             [0.01, 0.99],
             [0.01, 0.99],
             [0.99, 0.01],
             [0.01, 0.99],
             [0.99, 0.01],
             [0.01, 0.99]]
# trueValue = [[0, 1],
#              [0, 1],
#              [0, 1],
#              [0, 1],
#              [0, 1],
#              [1, 0],
#              [1, 0],
#              [1, 0],
#              [1, 0],
#              [1, 0]]

nn = NeuralNetwork([2, 3, 2], False, "data/testWeights.txt")
nn.trainNetwork(dataset, trueValue, 10000, 10, 0.5, 0, 0.000000000000001)
nn.testNetwork(dataset, trueValue, 0.99)