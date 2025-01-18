#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates the Schmid tensors for a given crystal type and returns as deviatoric 
5-vectors in the convention:
    1: sqrt(1/2)*(11-22)
    2: sqrt(3/2)*(33)
    3: sqrt(2)*(23)
    4: sqrt(2)*(13)
    5: sqrt(2)*(12)

INPUT:
ctype: Crystal type. Can be 'fcc', 'bcc', 'bct', or 'hcp'.
covera: A real scalar defining the c/a ratio for an HCP material. Default is the
    ideal ratio of 1.633, and is unused unless the crystal type is 'hcp'.

OUTPUT:
pdev: An array, shape [5, num_slip], containing the Schmid tensors as deviatoric
    5-vectors, where num_slip is the number of slip systems.

TO DO:
    - All caught up.

First created on Fri Jan 3 12:21:24 2025

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np
import sys
# import pandas as pd

def scys_calc_schmid_vecs(ctype, covera = 1.633):

    # Define n and d arrays
    if (ctype == 'bcc') or (ctype == 'fcc'):

        n = np.double(np.array([ \
            #110 slip
            [ 0,  1, -1], \
            [-1,  0,  1], \
            [ 1, -1,  0], \
            [ 0,  1, -1], \
            [ 1,  0,  1], \
            [ 1,  1,  0], \
            [ 0,  1,  1], \
            [ 1,  0,  1], \
            [-1,  1,  0], \
            [ 0,  1,  1], \
            [ 1,  0, -1], \
            [ 1,  1,  0], \
            # 112 slip for BCC
            # [-1, -1,  2], \
            # [-1,  2, -1], \
            # [ 2, -1, -1], \
            # [-1, -1, -2], \
            # [-1,  2,  1], \
            # [ 2, -1,  1], \
            # [-1,  1,  2], \
            # [-1, -2, -1], \
            # [ 2,  1, -1], \
            # [-1,  1, -2], \
            # [-1, -2,  1], \
            # [ 2,  1,  1] \
            ]))
        d = np.double(np.array([ \
            # 110 slip
            [ 1,  1,  1], \
            [ 1,  1,  1], \
            [ 1,  1,  1], \
            [-1,  1,  1], \
            [-1,  1,  1], \
            [-1,  1,  1], \
            [-1, -1,  1], \
            [-1, -1,  1], \
            [-1, -1,  1], \
            [ 1, -1,  1], \
            [ 1, -1,  1], \
            [ 1, -1,  1], \
            # 112 slip for BCC
            # [ 1,  1,  1], \
            # [ 1,  1,  1], \
            # [ 1,  1,  1], \
            # [ 1,  1, -1], \
            # [ 1,  1, -1], \
            # [ 1,  1, -1], \
            # [ 1, -1,  1], \
            # [ 1, -1,  1], \
            # [ 1, -1,  1], \
            # [ 1, -1, -1], \
            # [ 1, -1, -1], \
            # [ 1, -1, -1] \
            ]))
        
    elif (ctype == 'bct'):

        n = np.double(np.array([ \
            [ 1,  0,  0], \
            [ 0,  1,  0], \
            [ 1,  1,  0], \
            [ 1, -1,  0], \
            [ 1,  0,  0], \
            [ 0,  1,  0], \
            [ 1,  1,  0], \
            [ 1,  1,  0], \
            [ 1, -1,  0], \
            [ 1, -1,  0], \
            [ 1,  1,  0], \
            [ 1, -1,  0], \
            [ 1,  0,  0], \
            [ 1,  0,  0], \
            [ 0,  1,  0], \
            [ 0,  1,  0], \
            [ 0,  0,  1], \
            [ 0,  0,  1], \
            [ 0,  0,  1], \
            [ 0,  0,  1], \
            [ 1,  0,  1], \
            [ 1,  0, -1], \
            [ 0,  1,  1], \
            [ 0,  1, -1], \
            [ 1,  2,  1], \
            [ 1,  2,  1], \
            [-1, -2,  1], \
            [ 1, -2,  1], \
            [ 2,  1,  1], \
            [-2,  1,  1], \
            [-2, -1,  1], \
            [ 2, -1,  1] \
            ]))
        d = np.double(np.array([ \
            [ 0,  0,  1], \
            [ 0,  0,  1],\
            [ 0,  0,  1],\
            [ 0,  0,  1],\
            [ 0,  1,  0],\
            [ 1,  0,  0],\
            [ 1, -1,  1],\
            [-1,  1,  1],\
            [ 1,  1,  1],\
            [-1, -1,  1],\
            [-1,  1,  0],\
            [ 1,  1,  0],\
            [ 0,  1,  1],\
            [ 0,  1, -1],\
            [ 1,  0,  1],\
            [ 1,  0, -1],\
            [ 1,  1,  0],\
            [ 1, -1,  0],\
            [ 1,  1,  0],\
            [ 1, -1,  0],\
            [ 1,  0, -1],\
            [ 1,  0,  1],\
            [ 0,  1, -1],\
            [ 0,  1,  1],\
            [-1,  0,  1],\
            [ 1,  0,  1],\
            [ 1,  0,  1],\
            [-1,  0,  1],\
            [ 0, -1,  1],\
            [ 0, -1,  1],\
            [ 0,  1,  1],\
            [ 0,  1,  1] \
            ]))
        
    elif (ctype == 'hcp'):

        n = np.double(np.array([ \
            # Basal slip
            [ 0,  0,  0,  1], \
            [ 0,  0,  0,  1], \
            [ 0,  0,  0,  1], \
            # Prismatic slip
            [ 0,  1, -1,  0], \
            [-1,  0,  1,  0], \
            [ 1, -1,  0,  0], \
            # Pyramidal c+a slip
            [ 1,  0, -1,  1], \
            [ 1,  0, -1,  1], \
            [ 0,  1, -1,  1], \
            [ 0,  1, -1,  1], \
            [-1,  1,  0,  1], \
            [-1,  1,  0,  1], \
            [-1,  0,  1,  1], \
            [-1,  0,  1,  1], \
            [ 0, -1,  1,  1], \
            [ 0, -1,  1,  1], \
            [ 1, -1,  0,  1], \
            [ 1, -1,  0,  1], \
            # Pyramidal a slip
            # [ 1,  0, -1,  1], \
            # [ 0,  1, -1,  1], \
            # [-1,  1,  0,  1], \
            # [-1,  0,  1,  1], \
            # [ 0, -1,  1,  1], \
            # [ 1, -1,  0,  1] \
            ]))
        d = np.double(np.array([ \
            # Basal slip
            [ 2, -1, -1,  0], \
            [-1,  2, -1,  0], \
            [-1, -1,  2,  0], \
            # Prismatic slip
            [ 2, -1, -1,  0], \
            [-1,  2, -1,  0], \
            [-1, -1,  2,  0], \
            # Pyramidal c+a slip
            [-2,  1,  1,  3], \
            [-1, -1,  2,  3], \
            [-1, -1,  2,  3], \
            [ 1, -2,  1,  3], \
            [ 1, -2,  1,  3], \
            [ 2, -1, -1,  3], \
            [ 2, -1, -1,  3], \
            [ 1,  1, -2,  3], \
            [ 1,  1, -2,  3], \
            [-1,  2, -1,  3], \
            [-1,  2, -1,  3], \
            [-2,  1,  1,  3], \
            # Pyramidal a slip
            # [-1,  2, -1,  0], \
            # [ 2, -1, -1,  0], \
            # [-1, -1,  2,  0], \
            # [-1,  2, -1,  0], \
            # [ 2, -1, -1,  0], \
            # [-1, -1,  2,  0] \
            ]))
        
    else:

        sys.exit('Error: Crystal type not supported.')

    # Find number of slip systems
    N = np.shape(n)[0]
    
    # If HCP, first convert to Cartesian
    if (ctype == 'hcp'):
        ntrans = np.zeros([4, 3])
        ntrans[0, 0] = 1
        ntrans[0, 1] = 1 / np.sqrt(3)
        ntrans[1, 1] = 2 / np.sqrt(3) 
        ntrans[3, 2] = 1 / covera
        ntmp = np.matmul(n, ntrans)
        n = np.copy(ntmp)
        dtrans = np.zeros([4, 3])
        dtrans[0, 0] = 1.5
        dtrans[0, 1] = np.sqrt(3) / 2
        dtrans[1, 1] = np.sqrt(3)
        dtrans[3, 2] = covera
        dtmp = np.matmul(d, dtrans)
        d = np.copy(dtmp)

    # Normalize the slip normals and slip directions
    for i in range(0, N):
        n[i, :] = n[i, :] / np.linalg.norm(n[i, :])
        d[i, :] = d[i, :] / np.linalg.norm(d[i, :])

    # Define symmetric portions of the Schmid tensors
    pij = np.zeros([3, 3, N])
    for i in range(0, N):
        for j in range(0, 3):
            for k in range(0, 3):
                pij[j, k, i] = 0.5 * ((d[i, j] * n[i, k]) + (d[i, k] * n[i, j]))
                
    # Convert Schmid tensors to deviatoric 5-vectors
    pdev = np.zeros([5, N])
    for i in range(0, N):
        pdev[0, i] = np.sqrt(1/2) * (pij[0, 0, i] - pij[1, 1, i])
        pdev[1, i] = np.sqrt(3/2) * pij[2, 2, i]
        pdev[2, i] = np.sqrt(2) * pij[1, 2, i]
        pdev[3, i] = np.sqrt(2) * pij[0, 2, i]
        pdev[4, i] = np.sqrt(2) * pij[0, 1, i]

    # Print Schmid vectors to file for debugging
    # df = pd.DataFrame(np.transpose(pdev))
    # df.to_csv("/Users/mkasemer/downloads/schmids.csv")

    # Return pdev
    return pdev