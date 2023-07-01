# Simple python ReLU activation function implementation
# it takes an aritmetic array as input and gives the appropriate output
def ReLU(inputs):
    output = []
    for i in inputs:
        output.append(max(0,i))
    return(output)
