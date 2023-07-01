# Building a neural Network from scratch
# based on: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import math
import numpy as np
import matplotlib.pyplot as plt
# pip3 install nnfs

# use a specific seed to have the same values
np.random.seed(0)

E = math.e
exp_values = []

layer_outputs = [4.8, 1.21, 2.385]

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

class Layers_Dense:
    def __init__(self,n_inputs,n_neurons):
        # I set the weight to 0.9 instead of 0.1 in order to have better axis scaling in the graph
        self.weights = 0.9 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons ))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        # oneliner ReLU
        self.output = np.maximum(0, inputs)
