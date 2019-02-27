import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp = np.exp(L)
    suma =  exp.sum()
    return np.divide(exp, suma)