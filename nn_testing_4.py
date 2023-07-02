# Building a neural Network from scratch
# based on: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np
import matplotlib.pyplot as plt
# pip3 install nnfs
from nnfs.datasets.spiral import create_data

# use a specific seed to have the same values
np.random.seed(0)

class Layers_Dense:
    def __init__(self,n_inputs,n_neurons):
        # I set the weight to 0.9 instead of 0.1 in order to have better axis scaling in the graph
        self.weights = 0.9 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        # oneliner ReLU
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # We want sum of rows
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = create_data(samples=100, classes=3)

dense1 = Layers_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layers_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# print the first 5 outputs
print(activation2.output[:5])
