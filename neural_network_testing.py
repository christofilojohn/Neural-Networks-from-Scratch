# Building a neural Network from scratch
# This is a test to fit a sine wave with an array of neurons without an optimizer

import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# ReLU activation function testing
# method 1:
for i in inputs:
    if (i>0):
        output.append(i)
    else:
        output.append(0)
# method 2:
for i in inputs:
    output.append(max(0,i))


print(output)
class Layers_Dense:
    pass