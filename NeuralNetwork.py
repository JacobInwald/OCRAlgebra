import matplotlib.pyplot as plt
import utilities as util
import random
from tqdm import tqdm


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

        # This is an id that stores position in the nodes array
        self.number = number

    def feedForward(self):
        # This just sums the input nodes outputs and then passes that values through an activation function
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

    def generateNetwork(self, nodeNumbers):
        # This initialises all the variables for the network with different values
        # 1. Create a list of unconnected nodes with random biases and activations
        # 2. Add the input nodes and output nodes

        # Stage One
        index = -1
        for i in nodeNumbers:
            index += 1
            for x in range(i):
                # initialise nodes with None type inputs and outputs as well as biases and outputs
                self.nodes[index].append(Node([None], [None], random.uniform(0, 1), random.uniform(-10, 10), x))

        # Stage Two
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
        # This function saves the current state of the network to a text file
        # 1. Convert the data to a string and suitably seperate the different data types
        # 2. Put it into a text document

        # Initialisation
        stringToWrite = ""

        # Stage One
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

        # Stage Two
        file = open(filePath, 'w')
        file.write(stringToWrite)
        file.close()

    def loadFromFile(self, filePath):
        # This function opens a specified save file and puts the weights and biases within into the network
        # 1. Read the string from the file
        # 2. Split up the string into each of the individual weights and biases
        # 3. Replace the current weights and biases with the new ones

        # Initialisation and Stage One
        file = open(filePath, 'r')
        weights = file.read()
        file.close()

        # Stage Two
        weights = weights.split("|")
        for i in range(len(weights)):
            # This is specifically for the weight values
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

            # This is specifically for bias values
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

        # Stage Three
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
        # This function loads an input array into the neural network
        for i in range(len(inputArray)):
            self.nodes[0][i].output = inputArray[i]

    def getNewPaces(self, path):
        text = open(path)
        data = text.readlines()
        lp = float(data[0])
        mp = float(data[1])
        text.close()
        return lp, mp

    def feedForward(self):
        # This gets the output for each node based of the current input
        # 1. Loop through teh network
        # 2. Make each node feed forward
        # 3. Get the output for the output layers

        # Initialisation
        output = []

        # Stage One
        for i in range(len(self.nodes)):

            # skips first layer which already has their own outputs
            if i == 0:
                continue

            for node in self.nodes[i]:
                # Stage Two
                node.feedForward()

        # Stage Three
        for i in self.nodes[len(self.nodes) - 1]:
            output.append(i.output)

        return output

    def backPropagateCost(self, trueValue):
        # This function gets the error value for every node in the network
        # 1. Gets the output nodes error
        # 2. Loop through the whole network
        # 3. Calculate error for each node

        # Stage One
        for i in range(len(trueValue)):
            newCostDerivative = 2 * (trueValue[i] - self.nodes[self.layers - 1][i].output) * util.sigmoidDerivative(self.nodes[self.layers - 1][i].z)
            self.nodes[self.layers - 1][i].error = newCostDerivative

        # Stage Two
        for i in range(self.layers - 1, -1, -1):

            # Skips the last layer as it already has its error
            if i == self.layers - 1 or i == 0:
                continue

            for tgtNode in self.nodes[i]:
                cost = 0
                # Stage Three
                for x in tgtNode.outputNodes:
                    cost += x[0].error * x[1]
                tgtNode.error += cost * util.sigmoidDerivative(tgtNode.z)

    def updateWeights(self, learningPace):
        # Changes the weights based off error and learning pace
        # 1. Loop through the network
        # 2. Update weight and biases based of the back propagation algorithm

        # Stage One
        for i in range(self.layers):

            # This prevents errors because the input nodes have no input nodes themselves
            if i == 0:
                continue

            # Stage 2
            for tgtNode in self.nodes[i]:
                biasDelta = learningPace * tgtNode.error
                for x in tgtNode.inputNodes:
                    x[1] += biasDelta * x[0].output
                    x[0].outputNodes[tgtNode.number][1] = x[1]

                tgtNode.bias += biasDelta
                tgtNode.error = 0

    def trainNetwork(self, trainingData, trainingLabels, epochs, learningPace, lowestCost):
        # This trains the network to recognise numbers
        # 1. Loop through the training items
        # 2. Get the guess from the network
        # 3. Check guess with the correct answer and change the weights and biases connected to the answer
        # 4. Evaluate the cost and give user info

        # Initialisation
        costArray = []
        lp = learningPace

        # Stage One
        for x in range(epochs):
            costMean = 0
            for i in tqdm(range(len(trainingData))):

                # Stage Two
                self.loadInputs(trainingData[i])
                guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                guess = self.feedForward()

                # Stage Three
                trueValue = trainingLabels[i]
                self.backPropagateCost(trueValue)
                self.updateWeights(lp)

                # Stage Four
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
                    print("Guess:", guess)
                    print("Answer:", trueValue)
                    print("Learning Pace:", lp)
                    print("Loss:", costMean)
                    if costMean <= lowestCost and i != 0:
                        lowestCost = costMean
                        print("Saving weights ...")
                        self.saveToFile(self.path)
                    if costMean <= 0.01 and i != 0:
                        break
                    costMean = 0
                    print("-----------------------------------")

        plt.plot(costArray)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.show()

    def testNetwork(self, testData, testLabels, rightNumber):
        # This function runs the neural network for a test dataset in order for an answer to multiple inputs
        # 1. Loop through test inputs
        # 2. Get answer to the input
        # 3. Count the amount correct
        # 4. Calculate the percentages

        # Initialisation
        right = 0
        answer = -1
        numbersRight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Stage One
        for i in tqdm(range(len(testData))):

            # Stage Two
            self.loadInputs(testData[i])
            guess = self.feedForward()
            labels = testLabels[i].tolist()
            correct = labels.index(rightNumber)

            # Stage Three
            runningTotal = 0
            for x in guess:
                if x >= runningTotal:
                    runningTotal = x
                    answer = guess.index(x)
            if correct == answer:
                right += 1
                numbersRight[correct - 1] += 1

        # Stage Four
        print("Total percentage correct:", (right * 100) / len(testData), "%")
        for correct in numbersRight:
            print("Percentage correct for", numbersRight.index(correct) + 1, "is:",
                  (correct * 100) / (len(testData) / 10), "%")
        plt.show()

    def getAnswer(self, input):
        # This gets an answer for a specific input
        # 1. Protect the computer from inputs
        # 2. Feed forward for all nodes
        # 3. Get the output from the output nodes

        # Initialisation
        self.loadInputs(input)
        output = []

        for layer in self.nodes:

            # Stage One
            if self.nodes.index(layer) == 0:
                continue

            # Stage Two
            for node in layer:
                node.feedForward()

        # Stage Three
        for i in self.nodes[len(self.nodes) - 1]:
            output.append(i.output)
        x = 0
        for i in output:
            if i > x:
                x = i
        guess = output.index(x)

        return str(guess), str(x)

#
# dataset = [[2.7810836, 2.550537003],
#
#            [1.465489372, 2.362125076],
#
#            [3.396561688, 4.400293529],
#
#            [1.38807019, 1.850220317],
#
#            [3.06407232, 3.005305973],
#
#            [7.627531214, 2.759262235],
#
#            [5.332441248, 2.088626775],
#
#            [6.922596716, 1.77106367],
#
#            [8.675418651, -0.242068655],
#
#            [7.673756466, 3.508563011]]
#
# # trueValue = [[0.01, 0.99],
#
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
#
#
# nn = NeuralNetwork([2, 3, 2], False, "data/test.txt")
# nn.trainNetwork(dataset, trueValue, 10000, 1, 10)
# nn.testNetwork(dataset, trueValue, 1)