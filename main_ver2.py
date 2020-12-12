import random
import NeuralNetwork_ver2 as nn
import re
from sklearn.model_selection import train_test_split

file_name = "test_2.txt"
test = []
with open(file_name) as file:
    line = file.readline()
    while line:
        split_line = [val.strip(' ') for val in re.split(r'[()]', line.strip('(' + ')' + '\n'))]

        feature_vector = list(map(int, split_line[1].split(' ')))  # Convert string values to integer values
        label_val = int(split_line[2])  # Save Class Value

        # Append new Node object w/ label value, and the feature vector values
        test.append((label_val, feature_vector))

        line = file.readline()

training_set, testing_set = train_test_split(test, test_size=0.20, shuffle=True)
print("========TRAIN========")
for x in training_set:
   print(x[0], x[1])
print("========TEST========")
for x in testing_set:
    print(x[0], x[1])

neuralNetwork = nn.NeuralNetwork()

for i in range(10000):
    random.shuffle(training_set)
    for j in training_set:
        neuralNetwork.train(j[1], j[0])
    print(i)

for k in testing_set:
    temp = list(neuralNetwork.test(k[1]))
    print(k[0], '=====', temp, '=====', temp.index(max(temp)))
