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

    def __init__(self, inputNodes, outputNodes, output, number):
        #This will be a list of tuples, the first being the node the second being the weight
        self.inputNodes = inputNodes 
        self.outputNodes = outputNodes
        self.number = number
        #Sets activation function to sigmoid
        self.activationFunction = util.sigmoid
        self.output = output

    def feedForward(self):
        #This just sums the input nodes outputs and then passes that values through an activation function
        self.output = self.activationFunction(util.inputSummingFunction(self.inputNodes))


class NeuralNetwork:

    def __init__(self, nodeNumbers):
        #initialise variables
        self.nodes = [[] for _ in nodeNumbers]
        self.layers = len(nodeNumbers)
        self.generateNetwork(nodeNumbers)

    #generates a connected network based on an array of numbers e.g. [784, 16, 16, 10]
    def generateNetwork(self, nodeNumbers):
        #generates a void network with no connections but has the correct size
        
        index = -1
        for i in nodeNumbers:
            index += 1 
            for x in range(i):
                self.nodes[index].append(node([None], [None], random.uniform(0, 1), x))             


        #This loops through the generated list and sets the input nodes.
        for i in range(self.layers):
            layer = self.nodes[i]
            #To skip the first layer which is just inputs and are not actually nodes, so don't have input nodes.
            if(i == 0):
                continue
            #sets inputNodes
            for tgtNode in layer:
                #initialise inputNode as a None array with length of previous layer
                tgtNode.inputNodes = [None] * len(self.nodes[i-1])
                #loops through previous layer and sets weight and nodes
                for x in range(len(self.nodes[i-1])):
                    tgtNode.inputNodes[x] = [self.nodes[i-1][x], random.uniform(-1, 1)]
        

        #This loops through the generated list and sets the output nodes         
        for i in range(self.layers):
            layer = self.nodes[i]
            prevLayerIndex = i + 1
            #To skip the last layer which are the output nodes
            if(i == self.layers - 1):
                break
            
            #set outputNodes
            for tgtNode in layer:
                #initialises an empty array of length next layer
                tgtNode.outputNodes = [None] * len(self.nodes[i + 1])
                for x in range(len(self.nodes[i + 1])):
                    #loops trhough next layer and sets weights and nodes.
                    tgtNode.outputNodes[x] = [self.nodes[i + 1][x], self.nodes[i + 1][x].inputNodes[tgtNode.number][1]]
                
    def loadInputs(self, inputArray):
        #Loads an input array into the neural network
        for i in range(len(inputArray)):
            self.nodes[0][i].output = inputArray[i]

            
    def feedForward(self):
        output = []
        #does all the feed forwards for all the nodes
        for i in range(len(self.nodes)):
            if(i == 0):
                continue
            for node in self.nodes[i]:
                node.feedForward()
        #gets the output for all the output nodes
        for i in self.nodes[len(self.nodes) - 1]:
            output.append(i.output)
        return output


    def backPropogation(self, leastSquareEstimator):

        for i in range(self.layers):
            layer = self.nodes[i]
            #To skip the first layer which is just inputs and are not actually nodes, so don't have input nodes.
            if(i == 0):
                continue
            #sets inputNodes
            for tgtNode in layer:
                #initialise inputNode as a None array with length of previous layer
                #tgtNode.inputNodes = [None] * len(self.nodes[i-1])
                #loops through previous layer and sets weight and nodes
                for x in range(len(self.nodes[i-1])):
                    tgtNode.inputNodes[x][1] = tgtNode.inputNodes[x][1] + (leastSquareEstimator * 2)

        #This loops through the generated list and sets the output nodes         
        for i in range(self.layers):
            layer = self.nodes[i]
            prevLayerIndex = i + 1
            #To skip the last layer which are the output nodes
            if(i == self.layers - 1):
                break
            
            #set outputNodes
            for tgtNode in layer:
                #initialises an empty array of length next layer
                #tgtNode.outputNodes = [None] * len(self.nodes[i + 1])
                for x in range(len(self.nodes[i + 1])):
                    #loops trhough next layer and sets weights and nodes.
                    tgtNode.outputNodes[x][1] = self.nodes[i + 1][x].inputNodes[tgtNode.number][1]
        

    def costFunction(self, answer, trueValue):
        value = 0
        for i in range(len(answer)):
            value += (answer[i] - trueValue[i])**2
        return (value / 2)

    def leastSquareEstimator(self, answer, trueValue, costValue):
        nominator = 0
        denominator = 0
        for i in range(len(answer)):
            nominator += (answer[i] - trueValue[i]) * -(costValue - 0.01)
            denominator += (answer[i] - trueValue[i])**2
        value = nominator/denominator
        #value = value/abs(value)
        return value

    def trainNetwork(self, trainingData, trainingLabels):
        guess = []
        costMean = 0
        for i in range(len(trainingData)):
            #gets guess and true values
            self.loadInputs(trainingData[i])
            guess = self.feedForward()
            trueValue = trainingLabels[i]
            cost = self.costFunction(guess, trueValue)
            #costMean
            costMean += cost
            #gets gradient of graph
            leastSquareEstimator = self.leastSquareEstimator(guess, trueValue, cost)
            #performs backpropogation
            self.backPropogation(leastSquareEstimator)
            #gives update every 500 reps
            if(i % 500 == 0):
                print("Rep:", i)
                print("Actual:", trueValue)
                print("Guess:", guess)
                print("This cost:", cost)
                print("Cost:", costMean / 500)
                if(costMean <= 0.01):
                    return
                costMean = 0
                print("---------------------------------")
        print("Actual:", trueValue)
        print("Guess:", guess)
        print("Cost:", cost)
        print("---------------------------------")
            

    def giveAnswer(self, inputArray):
        #does all the stuff at once
        #self.loadInputs(inputArray)
        output = self.feedForward()
        answer = [0, 0]
        for i in output:
            if(i >= answer[1]):
                answer[0] = output.index(i)
                answer[1] = i
        return answer
            
    





            
##        print("input nodes")
##        for i in self.nodes:
##            print("next layer")
##            for x in i:
##                if(x.inputNodes != [None]):
##                    for y in x.inputNodes:
##                        print(y[0].output, ", ", y[1])
##                else:
##                    print("Nothing")
##                print("------------------")
##        print(" -------------------------")
##        print("output nodes")
##        for i in self.nodes:
##            print("next layer")
##            for x in i:
##                if(x.outputNodes != [None]):
##                    for y in x.outputNodes:
##                        print(y[0].output, ", ", y[1])
##                else:
##                    print("Nothing")
##                print("------------------")
##        
##
##nn = NeuralNetwork([784, 16, 16, 10])
##nn.loadInputs([0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.02164706,0.07988235,0.07988235,0.07988235
##,0.49917647,0.538,0.68941176,0.11094118,0.65447059,1.
##,0.96894118,0.50305882,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.12647059,0.14976471,0.37494118,0.60788235
##,0.67,0.99223529,0.99223529,0.99223529,0.99223529,0.99223529
##,0.88352941,0.67776471,0.99223529,0.94952941,0.76705882,0.25847059
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.20023529
##,0.934,0.99223529,0.99223529,0.99223529,0.99223529,0.99223529
##,0.99223529,0.99223529,0.99223529,0.98447059,0.37105882,0.32835294
##,0.32835294,0.22741176,0.16141176,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.07988235,0.86023529,0.99223529
##,0.99223529,0.99223529,0.99223529,0.99223529,0.77870588,0.71658824
##,0.96894118,0.94564706,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.32058824,0.61564706,0.42541176,0.99223529
##,0.99223529,0.80588235,0.05270588,0.01,0.17694118,0.60788235
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.06435294,0.01388235,0.60788235,0.99223529,0.35941176
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.54964706,0.99223529,0.74764706,0.01776471,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.05270588
##,0.74764706,0.99223529,0.28176471,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.14588235,0.94564706
##,0.88352941,0.63117647,0.42929412,0.01388235,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.32447059,0.94176471,0.99223529
##,0.99223529,0.472,0.10705882,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.18470588,0.73211765,0.99223529,0.99223529
##,0.59235294,0.11482353,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.07211765,0.37105882,0.98835294,0.99223529,0.736
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.97670588,0.99223529,0.97670588,0.25847059,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.18858824,0.51470588,0.72047059,0.99223529
##,0.99223529,0.81364706,0.01776471,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.16141176,0.58458824
##,0.89905882,0.99223529,0.99223529,0.99223529,0.98058824,0.71658824
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.10317647,0.45258824,0.868,0.99223529,0.99223529,0.99223529
##,0.99223529,0.79035294,0.31282353,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.09929412,0.26623529,0.83694118,0.99223529
##,0.99223529,0.99223529,0.99223529,0.77870588,0.32447059,0.01776471
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.07988235,0.67388235
##,0.86023529,0.99223529,0.99223529,0.99223529,0.99223529,0.76705882
##,0.32058824,0.04494118,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.22352941,0.67776471,0.88741176,0.99223529,0.99223529,0.99223529
##,0.99223529,0.95729412,0.52635294,0.05270588,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.538,0.99223529
##,0.99223529,0.99223529,0.83305882,0.53411765,0.52247059,0.07211765
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01,0.01,0.01
##,0.01,0.01,0.01,0.01])
##print(nn.feedForward())
##print(nn.giveAnswer([]))
##print(nn.costFunction(nn.feedForward(), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
