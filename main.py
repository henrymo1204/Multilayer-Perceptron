import random
import NeuralNetwork as nn
import re
from sklearn.model_selection import train_test_split

file_name = "test_2.txt"
test = []
# parses the text file and feeds them into a single list
with open(file_name) as file:
    line = file.readline()
    while line:
        split_line = [val.strip(' ') for val in re.split(r'[()]', line.strip('(' + ')' + '\n'))]
        # separates the value and the vector
        feature_vector = list(map(int, split_line[1].split(' ')))
        label_val = int(split_line[2])
        test.append((label_val, feature_vector))

        line = file.readline()

# Randomizes lists and places them into two lists 80/20 split
# Prints out the separate lists to ensure randomness
training_set, testing_set = train_test_split(test, test_size=0.20, shuffle=True)
print("========TRAIN========")
for x in training_set:
    print(x[0], x[1])
print("========TEST========")
for x in testing_set:
    print(x[0], x[1])

neuralNetwork = nn.NeuralNetwork()
# loops through 10k times
range_test = 10000
for i in range(range_test):
    random.shuffle(training_set)
    for j in training_set:
        neuralNetwork.train(j[1], j[0])
    if i == 1:
        print("Calculating...")
    if i == range_test/2:
        print("Half-way there...")
    if i == range_test*.75:
        print("Almost Done...")

# prints out and displays the testing set's results
for k in testing_set:
    temp = list(neuralNetwork.test(k[1]))
    print("Value: ", k[0], "Calculated Value: ", temp.index(max(temp)))
    print('=====', temp, '=====')
