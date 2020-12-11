import random
import numpy as np
import math


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


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
        self.outputNodes = 8
        self.weights_input_to_hidden = [[random.choice([-1, 1]) for x in range(self.inputNodes + 1)] for y in
                                        range(self.hiddenNodes + 1)]
        self.weights_hidden_to_output = [[random.choice([-1, 1]) for x in range(self.hiddenNodes + 2)] for y in
                                         range(self.outputNodes)]
        self.hidden_bias = 1
        self.output_bias = 1

    def train(self, vector, label):  # feed forward algorithm
        sigmoid_function = np.vectorize(sigmoid)  # sigmoid function to apply to the matrix
        derivative_sigmoid_function = np.vectorize(derivative_sigmoid)

        test = vector.copy()
        target = num_to_list(label)
        test += [self.hidden_bias]
        # feed forward from input to hidden
        inputs = np.array(test)
        # change inputs from horizontal form to vertical form
        inputs = inputs.reshape(11, 1)
        #print('input vector: ', inputs)
        hidden_weights = np.array(self.weights_input_to_hidden)
        #print('hidden weights: ', hidden_weights)
        hidden_result = hidden_weights.dot(inputs)
        #print('hidden results: ', hidden_result)
        hidden_result = sigmoid_function(hidden_result)
        # print('hidden result after applying sigmoid: ', hidden_result)
        hidden_result = np.append(hidden_result, [self.output_bias])
        # print('hidden result with output bias: ', hidden_result)

        # feed forward from hidden to output
        output_weights = np.array(self.weights_hidden_to_output)
        #print('output weights: ', output_weights)
        output_result = output_weights.dot(hidden_result)
        #print('output result: ', output_weights)
        result = sigmoid_function(output_result)
        #print('final result: ', result)
        guess = result

        # output error
        output_error = target - guess

        # output gradient
        guess = derivative_sigmoid_function(guess)
        output_gradient = guess.dot(output_error)
        output_gradient *= 0.1

        # hidden to output deltas
        transposed_hidden = np.transpose(hidden_result)
        weights_hidden_to_output_deltas = transposed_hidden * output_gradient
        # print(weights_hidden_to_output_deltas)
        # print(self.weights_hidden_to_output)
        current_weights_hidden_to_output = np.array(self.weights_hidden_to_output)
        deltas_weights_hidden_to_output = np.array(np.transpose(weights_hidden_to_output_deltas))
        self.weights_hidden_to_output = np.add(current_weights_hidden_to_output, deltas_weights_hidden_to_output)
        # print(self.weights_hidden_to_output)

        # hidden error
        transposed_input = np.transpose(self.weights_hidden_to_output)
        hidden_error = transposed_input.dot(output_error)
        #print(hidden_error)

        # hidden gradient
        hidden_gradient = derivative_sigmoid_function(hidden_result)
        hidden_gradient = hidden_gradient.dot(hidden_error)
        hidden_gradient *= 0.1

        # input to hidden deltas
        transposed_input = np.transpose(inputs)
        weights_input_to_hidden_deltas = transposed_input * hidden_gradient
        current_weights_input_to_hidden = np.array(self.weights_input_to_hidden)
        deltas_weights_input_to_hidden = np.array(np.transpose(weights_input_to_hidden_deltas))
        self.weights_input_to_hidden = np.add(current_weights_input_to_hidden, deltas_weights_input_to_hidden)

    def test(self, vector):  # testing algorithm
        sigmoid_function = np.vectorize(sigmoid)  # sigmoid function to apply to the matrix

        test = vector.copy()
        test += [self.hidden_bias]
        # feed forward from input to hidden
        inputs = np.array(test)
        # change inputs from horizontal form to vertical form
        inputs = inputs.reshape(11, 1)
        # print('input vector: ', inputs)
        hidden_weights = np.array(self.weights_input_to_hidden)
        # print('hidden weights: ', hidden_weights)
        hidden_result = hidden_weights.dot(inputs)
        # print('hidden results: ', hidden_result)
        hidden_result = sigmoid_function(hidden_result)
        # print('hidden result after applying sigmoid: ', hidden_result)
        hidden_result = np.append(hidden_result, [self.output_bias])
        # print('hidden result with output bias: ', hidden_result)

        # feed forward from hidden to output
        output_weights = np.array(self.weights_hidden_to_output)
        # print('output weights: ', output_weights)
        output_result = output_weights.dot(hidden_result)
        # print('output result: ', output_result)
        result = sigmoid_function(output_result)
        # print('final result: ', result)
        return result

#  temp = NeuralNetwork()
#  v = [1, 2]
#  temp.feedForward(v)
