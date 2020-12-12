import random
import NeuralNetwork_ver3 as nn

training_set = [[0, [1, 1]], [0, [0, 0]], [1, [1, 0]], [1, [0, 1]]]

neuralNetwork = nn.NeuralNetwork()

for i in range(5000):
    #random.shuffle(training_set)
    for j in training_set:
        neuralNetwork.train(j[1], j[0])
    print(i)

for k in training_set:
    print(k[0], '=====', neuralNetwork.test(k[1]))