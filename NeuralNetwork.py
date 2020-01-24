'''
    To Do:
        Generator Class
        Neuron Class
        Weight Class
        General Calculas class
'''
'import numpy as np'
import matplotlib.pyplot as plt
import utilities as util
import random


class node:

    def __init__(self, inputNodes, outputNodes, output, bias, number):
        # This will be a list of tuples, the first being the node the second being the weight
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.bias = bias
        self.number = number
        # Sets activation function to sigmoid
        self.activationFunction = util.sigmoid
        self.output = output
        self.error = 0

    def feedForward(self):
        # This just sums the input nodes outputs and then passes that values through an activation function
        self.output = self.activationFunction(self.sumInputs() + self.bias)

    def sumInputs(self):
        total = 0
        for i in self.inputNodes:
            total += i[1] * i[0].output
        return total


class NeuralNetwork:

    def __init__(self, nodeNumbers, loadFromFile, filePath):
        # initialise variables
        self.nodes = [[] for _ in nodeNumbers]
        self.layers = len(nodeNumbers)
        self.generateNetwork(nodeNumbers)
        self.path = filePath
        if not loadFromFile:
            self.saveToFile(self.path)
        else:
            self.loadFromFile(self.path)
        self.costArray = []

    # generates a connected network based on an array of numbers e.g. [784, 16, 16, self.layers]
    def generateNetwork(self, nodeNumbers):
        # generates a void network with no connections but has the correct size

        index = -1
        for i in nodeNumbers:
            index += 1
            for x in range(i):
                # initialise nodes with biases and outputs
                self.nodes[index].append(node([None], [None], random.uniform(0, 1), random.uniform(-10, 10), x))

                # This loops through the generated list and sets the input nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the first layer which is just inputs and are not actually nodes, so don't have input nodes.
            if (i == 0):
                continue
            # sets inputNodes
            for tgtNode in layer:
                # initialise inputNode as a None array with length of previous layer
                tgtNode.inputNodes = [None] * len(self.nodes[i - 1])
                # loops through previous layer and sets weight and nodes
                for x in range(len(self.nodes[i - 1])):
                    tgtNode.inputNodes[x] = [self.nodes[i - 1][x], random.uniform(-1, 1)]

        # This loops through the generated list and sets the output nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            prevLayerIndex = i + 1
            # To skip the last layer which are the output nodes
            if (i == self.layers - 1):
                break

            # set outputNodes
            for tgtNode in layer:
                # initialises an empty array of length next layer
                tgtNode.outputNodes = [None] * len(self.nodes[i + 1])
                for x in range(len(self.nodes[i + 1])):
                    # loops trhough next layer and sets weights and nodes.
                    tgtNode.outputNodes[x] = [self.nodes[i + 1][x], self.nodes[i + 1][x].inputNodes[tgtNode.number][1]]

    def saveToFile(self, filePath):
        stringToWrite = ""
        for i in range(len(self.nodes)):
            layers = self.nodes[i]
            if (i == 0):
                continue
            stringToWrite += "#"
            for node in layers:
                stringToWrite += "~"
                for data in node.inputNodes:
                    stringToWrite += ","
                    stringToWrite += str(data[1])
        stringToWrite += "|"
        for layers in self.nodes:
            stringToWrite += "#"
            for node in layers:
                stringToWrite += "~"
                stringToWrite += str(node.bias)

        file = open(filePath, 'w')
        file.write(stringToWrite)
        file.close()

    def loadFromFile(self, filePath):
        file = open(filePath, 'r')
        weights = file.read()
        file.close()
        weights = weights.split("|")

        for i in range(len(weights)):
            if (i == 0):
                weights[i] = weights[i].split("#")
                for x in range(len(weights[i])):
                    weights[i][x] = weights[i][x].split("~")
                    for y in range(len(weights[i][x])):
                        weights[i][x][y] = weights[i][x][y].split(",")
                        while '' in weights[i][x][y]:
                            weights[i][x][y].remove('')
                    while '' in weights[i][x]:
                        weights[i][x].remove('')
                while '' in weights[i]:
                    weights[i].remove('')
            if (i == 1):
                weights[i] = weights[i].split("#")
                for x in range(len(weights[i])):
                    weights[i][x] = weights[i][x].split("~")
                    while '' in weights[i][x]:
                        weights[i][x].remove('')
                while '' in weights[i]:
                    weights[i].remove('')

        weights = [x for x in weights if x]

        for i in range(len(weights)):
            if (i == 0):
                for x in range(len(weights[i])):
                    if (x == 0):
                        continue
                    weights[i][x] = [a for a in weights[i][x] if a]
                    for y in range(len(weights[i][x])):
                        for z in range(len(weights[i][x][y])):
                            self.nodes[x][y].inputNodes[z][1] = float(weights[i][x][y][z])
            if (i == 1):
                weights[i] = [a for a in weights[i] if a]
                for x in range(len(weights[i])):
                    weights[i][x] = [a for a in weights[i][x] if a]
                    for y in range(len(weights[i][x])):
                        self.nodes[x][y].bias = float(weights[i][x][y])

        for i in range(self.layers):
            layer = self.nodes[i]
            prevLayerIndex = i + 1
            # To skip the last layer which are the output nodes
            if (i == self.layers - 1):
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

    def feedForward(self):
        output = []
        # does all the feed forwards for all the nodes
        for i in range(len(self.nodes)):
            # skips first layer which already has their own outputs
            if i == 0:
                continue

            for node in self.nodes[i]:
                node.feedForward()
        # gets the output for all the output nodes
        for i in self.nodes[len(self.nodes) - 1]:
            output.append(i.output)
        return output

    def backPropogateCost(self, answer, trueValue):
        # Initialise cost Matrix with the output layer first
        lastLayer = util.costFunctionOutputLayer(answer, trueValue)
        for i in range(len(lastLayer)):
            self.nodes[self.layers - 1][i].error = lastLayer[i]
        # loops through the nodes array to propogate backwards for the error
        for i in range(self.layers - 1, -1, -1):

            if (i == self.layers - 1):
                continue

            for tgtNode in self.nodes[i]:
                cost = 0
                for x in tgtNode.outputNodes:
                    cost += x[1] * x[0].error
                tgtNode.error = cost

    def updateWeights(self, learningPace):
        # This loops through the generated list and sets the input nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            # To skip the first layer which is just inputs and are not actually nodes, so don't have input nodes.
            if (i == 0):
                continue
            # updates weights
            for tgtNode in layer:
                # loops through previous layer and sets weight and nodes
                for x in tgtNode.inputNodes:
                    x[1] += learningPace * tgtNode.error * x[0].output
                tgtNode.bias += learningPace * tgtNode.error
        # This loops through the generated list and sets the output nodes
        for i in range(self.layers):
            layer = self.nodes[i]
            prevLayerIndex = i + 1
            # To skip the last layer which are the output nodes
            if (i == self.layers - 1):
                break

            # set outputNodes
            for tgtNode in layer:
                for x in range(len(self.nodes[i + 1])):
                    # loops through next layer and sets weights and nodes.
                    tgtNode.outputNodes[x][1] = self.nodes[i + 1][x].inputNodes[tgtNode.number][1]

    def evaluateCost(self, answer, trueValue):
        value = 0
        for i in range(len(answer)):
            value += (answer[i] - trueValue[i]) ** 2
        return value

    def trainNetwork(self, trainingData, trainingLabels, epochs, learningPace):
        guess = []
        costMean = 0
        for x in range(epochs):
            costMean = 0
            for i in range(len(trainingData)):
                # gets guess and true values
                self.loadInputs(trainingData[i])
                guess = self.feedForward()
                trueValue = trainingLabels[i]
                self.backPropogateCost(guess, trueValue)
                self.updateWeights(learningPace)
                cost = self.evaluateCost(guess, trueValue)
                costMean += cost

                if(i % 5000 == 0):
                        if(i != 0):
                            self.costArray.append(costMean / 5000)
                        else:
                            self.costArray.append(costMean)
                        print("Epoch:", x)
                        print("Rep:", i)
                        print("Guess:", guess)
                        print("Answer:", trueValue)
                        print("Cost:", costMean / 5000)
                        if(costMean / 5000 <= 0.05 and i != 0):
                            self.saveToFile()
                            return
                        costMean = 0
                        print("Saving weights ...")
                        self.saveToFile(self.path)
                        print("-----------------------------------")
            if costMean / len(trainingData) >= 0.5:
                break
            if costMean / len(trainingData) <= 0.05:
                self.saveToFile()
                break

    def testNetwork(self, testData, testLabels):
        right = 0
        for i in range(len(testData)):
            self.loadInputs(testData[i])
            guess = self.feedForward()
            trueValue = testLabels[i]
            correct = int(trueValue[0])
            runningTotal = 0
            for i in guess:
                if i >= runningTotal:
                    runningTotal = i
                    answer = guess.index(i)
            # for i in trueValue:
            #     if i >= runningTotal:
            #         runningTotal = i
            #         correct = trueValue.index(i)

            if correct == answer:
                right += 1
        print("Percentage correct:", (right * 100) / len(testData), "%")


    def giveAnswer(self, inputArray):
        # does all the stuff at once
        self.loadInputs(inputArray)
        output = self.feedForward()
        answer = [0, 0]
        for i in output:
            if (i >= answer[1]):
                answer[0] = output.index(i)
                answer[1] = i
        return answer
#
#
# dataset = [[2.7810836, 2.550537003],
#            [1.465489372, 2.362125076],
#            [3.396561688, 4.400293529],
#            [1.38807019, 1.850220317],
#            [3.06407232, 3.005305973],
#            [7.627531214, 2.759262235],
#            [5.332441248, 2.088626775],
#            [6.922596716, 1.77106367],
#            [8.675418651, -0.242068655],
#            [7.673756466, 3.508563011]]
# # trueValue = [[0.01, 0.99],
# #              [0.01, 0.99],
# #              [0.01, 0.99],
# #              [0.01, 0.99],
# #              [0.01, 0.99],
# #              [0.99, 0.01],
# #              [0.99, 0.01],
# #              [0.99, 0.01],
# #              [0.99, 0.01],
# #              [0.99, 0.01]]
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
# # count = 10
# # costMean = 0
# # costArray = []
# # nn = NeuralNetwork([2, 3, 2], False)
# # good = False
# # while not good:
# #     nn = NeuralNetwork([2, 3, 2], False, "data/weights.txt")
# #     nn.trainNetwork(dataset, trueValue, 10, 0.5)
# #     for i in dataset:
# #         nn.loadInputs(i)
# #         costMean += nn.evaluateCost(nn.feedForward(), trueValue[dataset.index(i)])
# #     costArray.append(costMean)
# #     if costMean <= 0.05:
# #         nn.saveToFile()
# #         good = True
# #     costMean = 0
# #
# # plt.plot(costArray)
# # plt.ylabel("Cost")
# # plt.xlabel("Attempts")
# # plt.show()
#
# nn = NeuralNetwork([2, 3, 2], False, "data/weights.txt")
# nn.trainNetwork(dataset, trueValue, 100000, 1)
# nn.testNetwork(dataset, trueValue)