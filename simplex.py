
import numpy as np

"""
A, b, c

A = [[ 1.  1.  1.  1.  1.  0.  0.]
    [ 2.  1. -1. -1.  0. -1.  0.]
    [ 0. -1.  0.  1.  0.  0. -1.]],
b = [40 10 12],
c = [-0.5 -3.  -1.  -4.   0.   0.   0. ])
"""

def simplex(A: np.array, b: np.array, c: np.array):
