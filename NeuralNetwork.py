import random
import numpy as np
import math


# sigmoid function
def sigmoid(x):
    if x < 0:
        return 1.0 - 1.0 / (1.0 + np.exp(x))
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1.0 - x)


def num_to_list(x):
    if x == 0:
        return [1, 0, 0, 0, 0, 0, 0, 0]
    elif x == 1:
        return [0, 1, 0, 0, 0, 0, 0, 0]
    elif x == 2:
        return [0, 0, 1, 0, 0, 0, 0, 0]
    elif x == 3:
        return [0, 0, 0, 1, 0, 0, 0, 0]
    elif x == 4:
        return [0, 0, 0, 0, 1, 0, 0, 0]
    elif x == 5:
        return [0, 0, 0, 0, 0, 1, 0, 0]
    elif x == 6:
        return [0, 0, 0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 1]


class NeuralNetwork:
    def __init__(self):  # constructor
        self.inputNodes = 10
        self.hiddenNodes = 10
        self.outputNodes = 1
        self.weights_input_to_hidden = [[random.choice([-1, 1]) for x in range(self.inputNodes)] for y in
                                        range(self.hiddenNodes)]
        self.weights_hidden_to_output = [[random.choice([-1, 1]) for x in range(self.hiddenNodes)] for y in
                                         range(self.outputNodes)]
        self.hidden_bias = [[1 for x in range(1)] for y in range(self.hiddenNodes)]
        self.output_bias = [[1 for x in range(1)] for y in range(self.outputNodes)]

    def train(self, vector, label):  # feed forward algorithm
        sigmoid_function = np.vectorize(sigmoid)  # sigmoid function to apply to the matrix
        derivative_sigmoid_function = np.vectorize(derivative_sigmoid)

        test = vector.copy()
        target = num_to_list(label)
        target = np.array(target)
        target = target.reshape(8, 1)
        inputs = np.array(test)
        inputs = inputs.reshape(10, 1)
        inputs = inputs / 100
        # print('input vector: ', inputs)
        hidden_weights = np.array(self.weights_input_to_hidden)
        # print('hidden weights: ', hidden_weights)
        hidden_result = hidden_weights.dot(inputs)
        # print('hidden results: ', hidden_result)
        hidden_result = np.add(hidden_result, self.hidden_bias)
        # print('hidden result with output bias: ', hidden_result)
        hidden_result = sigmoid_function(hidden_result)
        # print('hidden result after applying sigmoid: ', hidden_result)

        # feed forward from hidden to output
        output_weights = np.array(self.weights_hidden_to_output)
        # print('output weights: ', output_weights)
        output_result = output_weights.dot(hidden_result)
        # print('output result: ', output_weights)
        output_result = np.add(output_result, self.output_bias)
        result = sigmoid_function(output_result)
        result = np.array(result)
        # print(result)

        # output error
        output_error = np.subtract(target, result)
        # print(output_error)

        # output gradient
        result = derivative_sigmoid_function(result)
        output_gradient = np.multiply(output_error, result)
        output_gradient *= 0.1

        # print(output_gradient)

        # hidden to output deltas
        transposed_hidden = np.transpose(hidden_result)
        # print(transposed_hidden)
        weights_hidden_to_output_deltas = output_gradient.dot(transposed_hidden)
        # print(weights_hidden_to_output_deltas)
        # print(self.weights_hidden_to_output)
        current_weights_hidden_to_output = np.array(self.weights_hidden_to_output)
        deltas_weights_hidden_to_output = np.array(weights_hidden_to_output_deltas)
        self.weights_hidden_to_output = np.add(current_weights_hidden_to_output, deltas_weights_hidden_to_output)
        output_bias = np.array(self.output_bias)
        self.output_bias = np.add(output_bias, output_gradient)
        # print(self.weights_hidden_to_output)

        # hidden error
        transposed_input = np.transpose(self.weights_hidden_to_output)
        hidden_error = transposed_input.dot(output_error)
        #print(hidden_error)

        # hidden gradient
        hidden_gradient = derivative_sigmoid_function(hidden_result)
        hidden_gradient = np.multiply(hidden_error, hidden_gradient)
        hidden_gradient *= 0.1

        # input to hidden deltas
        transposed_input = np.transpose(inputs)
        weights_input_to_hidden_deltas = hidden_gradient.dot(transposed_input)
        current_weights_input_to_hidden = np.array(self.weights_input_to_hidden)
        deltas_weights_input_to_hidden = np.array(weights_input_to_hidden_deltas)
        self.weights_input_to_hidden = np.add(current_weights_input_to_hidden, deltas_weights_input_to_hidden)
        hidden_bias = np.array(self.hidden_bias)
        self.hidden_bias = np.add(hidden_bias, hidden_gradient)

    def test(self, vector):  # testing algorithm
        sigmoid_function = np.vectorize(sigmoid)  # sigmoid function to apply to the matrix

        test = vector.copy()
        inputs = np.array(test)
        inputs = inputs.reshape(10, 1)
        inputs = inputs / 100
        # print('input vector: ', inputs)
        hidden_weights = np.array(self.weights_input_to_hidden)
        # print('hidden weights: ', hidden_weights)
        hidden_result = hidden_weights.dot(inputs)
        # print('hidden results: ', hidden_result)
        hidden_result = np.add(hidden_result, self.hidden_bias)
        # print('hidden result with hidden bias: ', hidden_result)
        hidden_result = sigmoid_function(hidden_result)
        # print('hidden result after applying sigmoid: ', hidden_result)

        # feed forward from hidden to output
        output_weights = np.array(self.weights_hidden_to_output)
        # print('output weights: ', output_weights)
        output_result = output_weights.dot(hidden_result)
        # print('output result: ', output_result)
        output_result = np.add(output_result, self.output_bias)
        result = sigmoid_function(output_result)
        return result
