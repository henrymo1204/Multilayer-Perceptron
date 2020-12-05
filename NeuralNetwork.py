import random
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NeuralNetwork:
    def __init__(self):  # constructor
        self.inputNodes = 2
        self.hiddenNodes = 2
        self.outputNodes = 1
        self.weights_input_to_hidden = [[random.choice([-1, 1]) for x in range(self.inputNodes)] for y in
                                        range(self.hiddenNodes)]
        self.weights_hidden_to_output = [[random.choice([-1, 1]) for x in range(self.hiddenNodes)] for y in
                                         range(self.outputNodes)]
        self.hidden_bias = [[random.choice([-1, 1]) for x in range(1)] for y in range(self.hiddenNodes)]
        self.output_bias = [[random.choice([-1, 1]) for x in range(1)] for y in range(self.outputNodes)]

    def feedForward(self, vector):  # feed forward algorithm
        sigmoid_function = np.vectorize(sigmoid)  # sigmoid function to apply to the matrix

        # feed forward from input to hidden
        inputs = np.array(vector)
        # change inputs from horizontal form to vertical form
        inputs = inputs.reshape(2, 1)
        print('input vector: ', inputs)
        hidden_weights = np.array(self.weights_input_to_hidden)
        print('hidden weights: ', hidden_weights)
        hidden_result = hidden_weights.dot(inputs)
        print('hidden results: ', hidden_result)
        hidden_result += self.hidden_bias
        print('hidden result after adding bias: ', hidden_result)
        hidden_result = sigmoid_function(hidden_result)
        print('hidden result after applying sigmoid: ', hidden_result)

        # feed forward from hidden to output
        output_weights = np.array(self.weights_hidden_to_output)
        print('output weights: ', output_weights)
        output_result = output_weights.dot(hidden_result)
        print('output result: ', output_weights)
        output_result += self.output_bias
        print('output result after adding bias: ', output_result)
        result = sigmoid_function(output_result)
        print('final result: ', result[0][0])

        # return result[0][0]

    # def backPropagation(self):  # back propagation algorithm

    # def train(self, inputs, target):  #training algorithm

    # def test(self):  # testing algorithm

#  temp = NeuralNetwork()
#  v = [1, 2]
#  temp.feedForward(v)