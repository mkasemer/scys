#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates the transformation of a deviatoric tensor from a frame/basis "A" to a
    secondary frame/basis "B". The tensors are stored as 5-vectors, in the 
    convention:
    1: sqrt(1/2)*(11-22)
    2: sqrt(3/2)*(33)
    3: sqrt(2)*(23)
    4: sqrt(2)*(13)
    5: sqrt(2)*(12)

INPUT:
vec: A vector, shape [5, ], storing a deviatoric tensor as a 5-vector in the
    above convention. The tensor is described in an initial frame/basis, "A".
a: An array, shape [3, 3], storing a matrix describing the transformation from
    frame "A" to a secondary frame, "B".

OUTPUT:
vec_trans: A vector, shape [5, ], storing a deviatoric tensor as a 5-vector in
    the above convention. The tensor is described in the transformed 
    frame/basis, "B", and is found via:
    vec_trans = Q*vec

First created on Sun Dec 22 12:24:39 2024

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np

def scys_calc_basis_transform_5vec(vec, a):

    # Define 5x5 transformation matrix
    Q = np.zeros([5, 5])
    Q[0, 0] = 0.5 * (a[0, 0]**2 - a[0, 1]**2 - a[1, 0]**2 + a[1, 1]**2)
    Q[0, 1] = (np.sqrt(3)/2) * (a[0, 2]**2 - a[1, 2]**2)
    Q[0, 2] = (a[0, 1] * a[0, 2]) - (a[1, 1] * a[1, 2])
    Q[0, 3] = (a[0, 0] * a[0, 2]) - (a[1, 0] * a[1, 2])
    Q[0, 4] = (a[0, 1] * a[0, 1]) - (a[1, 0] * a[1, 1])
    Q[1, 0] = (np.sqrt(3)/2) * (a[2, 0]**2 - a[2, 1]**2)
    Q[1, 1] = (1.5 * a[2, 2]**2) - 0.5
    Q[1, 2] = np.sqrt(3) * a[2, 1] * a[2, 2]
    Q[1, 3] = np.sqrt(3) * a[2, 0] * a[2, 2]
    Q[1, 4] = np.sqrt(3) * a[2, 0] * a[2, 1]
    Q[2, 0] = (a[1, 0] * a[2, 0]) - (a[1, 1] * a[2, 1])
    Q[2, 1] = np.sqrt(3) * a[1, 2] * a[2, 2]
    Q[2, 2] = (a[1, 1] * a[2, 2]) - (a[1, 2] * a[2, 1])
    Q[2, 3] = (a[1, 0] * a[2, 2]) - (a[2, 0] * a[1, 2])
    Q[2, 4] = (a[1, 0] * a[2, 1]) - (a[1, 1] * a[2, 0])
    Q[3, 0] = (a[0, 0] * a[2, 0]) - (a[0, 1] * a[2, 1])
    Q[3, 1] = np.sqrt(3) * a[0, 2] * a[2, 2]
    Q[3, 2] = (a[0, 1] * a[2, 2]) - (a[0, 2] * a[2, 1])
    Q[3, 3] = (a[0, 0] * a[2, 2]) - (a[0, 2] * a[2, 0])
    Q[3, 4] = (a[0, 0] * a[2, 1]) - (a[0, 1] * a[2, 0])
    Q[4, 0] = (a[0, 0] * a[1, 0]) - (a[0, 1] * a[1, 1])
    Q[4, 1] = np.sqrt(3) * a[0, 2] * a[1, 2]
    Q[4, 2] = (a[0, 1] * a[1, 2]) - (a[0, 2] * a[1, 1])
    Q[4, 3] = (a[0, 0] * a[1, 2]) - (a[0, 2] * a[1, 0])
    Q[4, 4] = (a[0, 0] * a[1, 1]) - (a[0, 1] * a[1, 0])

    # Transform to new frame
    vec_trans = np.matmul(Q, vec)

    # Return as output
    return vec_trans