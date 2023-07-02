import numpy as np

layer_outputs = [4.8, 1.21, 2.385]
exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

print(exp_values)
print(norm_values)
print("Almost 1: ", sum(norm_values))