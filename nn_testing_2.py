# Building a neural Network from scratch

# based on: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np
# pip3 install nnfs
from nnfs.datasets.spiral import create_data
# use a specific seed to have the same values
np.random.seed(0)

X, y = create_data(100, 3)

class Layers_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons ))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        # oneliner ReLU
        self.output = np.maximum(0, inputs)

# We have 2 unique features that describe our dataset (X, y) so n_inputs = 2
layer1 = Layers_Dense(2,5)
layer1.forward(X)
activation1 = Activation_ReLU()
activation1.forward(layer1.output)

# Testing
#print(layer1.output)
print("Only positive data because of ReLU:\n",activation1.output)