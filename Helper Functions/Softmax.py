# Simple python Softmax activation function implementation
# it takes an aritmetic array as input and gives the appropriate output
import numpy as np

def Softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # We want sum of rows
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return(probabilities)