import numpy as np
import math

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    cross_entropy = 0.0
    for y, p in zip(Y, P):
        cross_entropy = cross_entropy - y*math.log(p) - (1-y)*math.log(1-p)
    return cross_entropy