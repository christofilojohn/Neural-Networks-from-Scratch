# Simple python ReLU activation function implementation
def ReLU(inputs, method2=True):
    output = []
    if(not method2): 
        # 2 equal methodologies for testing
        # method 1:
        for i in inputs:
            if (i>0):
                output.append(i)
            else:
                output.append(0)       
    else:
        # method 2:
        for i in inputs:
            output.append(max(0,i))
    return(output)

# example usage
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = ReLU(inputs)
print("Method1: ",output)
output = ReLU(inputs,False)
print("Method2: ",output)
