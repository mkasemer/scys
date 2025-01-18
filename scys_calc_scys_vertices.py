#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates the deviatoric stresses that represent the vertices of the single 
crystal yield surface.

INPUT:
ctype: Crystal type. Can be 'fcc', 'bcc', 'bct', or 'hcp'.
tau: An array, shape [nslip, 2], containing the magnitudes of the critical
    resolved shear stress (strength) values for each slip system in both 
    "positive" (first column) and "negative" slip (second column). Do not
    include negative values in either column - sign is handled internally.
covera: A real scalar defining the c/a ratio for an HCP material. Default is the
    ideal ratio of 1.633, and is unused unless the crystal type is 'hcp'.

OUTPUT:
verts: An array, shape [5, MV], containing the deviatoric stress vertices 
    that comprise the single crystal yield surface, where MV is the number of 
    vertices. Each vector is a devatoric stress where the components are stored 
    in a 5-vector form, following the convention:
    1: sqrt(1/2)*(11-22)
    2: sqrt(3/2)*(33)
    3: sqrt(2)*(23)
    4: sqrt(2)*(13)
    5: sqrt(2)*(12)
MV: The number of vertices.
theta_bar: The mean of the angle to the nearest neighbor among all vertices.

TO DO:
    - All caught up.

First created on Sun Aug 20 20:21:22 2023

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np
from itertools import combinations
# import pandas as pd 
import sys
from scys_calc_theta_bar import scys_calc_theta_bar
from scys_calc_schmid_vecs import scys_calc_schmid_vecs

def scys_calc_scys_vertices(ctype, tau, covera = 1.633):

    # Define a numerical tolerance for calculations
    tol = 1e-4
        
    # Calculate Schmid tensors as deviatoric 5-vectors
    pdev = scys_calc_schmid_vecs(ctype, covera)
    N = np.shape(pdev)[1]

    # Ensure that n, d, tau have the same number of slip systems
    if (np.shape(tau)[0] != N):
        sys.exit('Error: n, d, and tau do not have same N.')
    
    # Calculate all combinations of +/- unit CRSS (scaled by tau values later)
    tauperms = np.empty([5, 32])
    for i in range(0, 32):
        permstr = bin(i)[2: ]
        l = len(permstr)
        permstr = str(0) * (5 - l) + permstr
        for j in range(0, 5):
            tauperms[j, i] = float(permstr[j])
    tauperms[tauperms == 0] = -1
    ntauperms = np.shape(tauperms)[1]
        
    # Find the unique combinations of 5 slip systems
    combos = np.array(list(combinations(range(0, N), 5)))
    M = np.shape(combos)[0]
    
    # Solve for stresses which simultaneously satisfy CRSS for 5 independent 
    # slip systems simultaneously
    Pij = np.zeros([5, 5])
    detPij = np.zeros([M, ])
    verts = np.zeros([5, M, ntauperms])
    for i in range(0, M):
        
        # Construct 5x5 matrix of unique combination of Schmid tensors
        Pij[0, :] = pdev[:, combos[i, 0]]
        Pij[1, :] = pdev[:, combos[i, 1]]
        Pij[2, :] = pdev[:, combos[i, 2]]
        Pij[3, :] = pdev[:, combos[i, 3]]
        Pij[4, :] = pdev[:, combos[i, 4]]
        
        # Find the determinant of this matrix to determine linear independence
        detPij[i] = np.linalg.det(Pij[:, :])
        
        # If detPij is nonzero, find stress which satisfies all CRSS values
        if (abs(detPij[i]) > tol):
            
            # Construct an array of CRSS values for the 5 slip systems for all 
            # 32 permutations of the signed CRSS values (allows +/- anisotropy)
            Tau = np.zeros(np.shape(tauperms))
            # Loop over the 5 slip systems
            for j in range (0, 5):
                # Loop over the number of +/- permutations
                for k in range (0, ntauperms):
                    # If the value is positive, gather from first col of tau
                    if tauperms[j, k] == 1:
                        Tau[j, k] = tauperms[j, k] * tau[combos[i, j], 0]
                    # If the value is negative, gather from second col of tau
                    if tauperms[j, k] == -1:
                        Tau[j, k] = tauperms[j, k] * tau[combos[i, j], 1]
            
            # Solve for the stress state to satisfy M*verts=Tau. This will
            # handle all 32 vectors of tau in Tau simultaneously.
            verts[:, i, :] = np.linalg.solve(Pij, Tau)
                    
            # As an extra check of the solver above: for a given stress state, 
            # make sure that M*verts=Tau. This is likely overkill and only 
            # necessary for debugging, it may be fine to comment out to speed up 
            # code.
            # if np.any(abs(np.matmul(Pij, verts[:, i, :]) - Tau) > tol):
            #     print('Warning: Invalid vertex!')
                
    # Remove linearly dep values from verts array, reshape, find unique values.
    # What's left are the candidate vertices. These vertices satisfy the CRSS
    # condition for 5 linearly independent slip systems simultaneously.
    verts = verts[:, (abs(detPij) > tol), :]
    verts = verts.reshape((5, np.shape(verts)[1] * np.shape(verts)[2]))
    verts = np.unique(verts.round(decimals = 4), axis = 1)
    MC = np.shape(verts)[1]
    
    # Calculate whether a candidate vertex resides on the inner convex hull
    h = np.zeros([MC, 2*N])
    for i in range(0, N):
        
        # Find a stress point on a facet and the negative facet
        stress_point = np.zeros([5,])
        if (pdev[0, i] != 0):
            stress_point[0] = (tau[i, 0] / pdev[0, i])
        elif (pdev[1, i] != 0):
            stress_point[1] = (tau[i, 0] / pdev[1, i])
        elif (pdev[2, i] != 0):
            stress_point[2] = (tau[i, 0] / pdev[2, i])
        elif (pdev[3, i] != 0):
            stress_point[3] = (tau[i, 0] / pdev[3, i])
        elif (pdev[4, i] != 0):
            stress_point[4] = (tau[i, 0] / pdev[4, i])

        for j in range(0, MC):
            
            # Find dot product between difference vector and the plane normal
            h[j, i] = np.dot((verts[:, j] - stress_point), pdev[:, i])
            h[j, i+N] = np.dot((verts[:, j] - \
                (-stress_point * (tau[i, 1] / tau[i, 0]))), -pdev[:, i])
        
    # Stress vertices are those in which all dot products are less than or 
    # equal to zero - i.e., vertices which are either inside or on every
    # individual facet
    vertrows = np.sum(h <= tol, axis = 1)
    verts = verts[:, vertrows == (2 * N)]

    # Print vertices to file for debugging
    # df = pd.DataFrame(np.transpose(verts))
    # df.to_csv("/Users/mkasemer/downloads/verts.csv")

    # Caclulate mean angle to the nearest neighbor
    theta_bar = scys_calc_theta_bar(verts)

    # Print pertinent values to terminal
    print("Stress vertices", verts)
    MV = np.shape(verts)[1]
    print("Number of vertices:", MV)
    print("Mean angle to nearest neighbor:", theta_bar)

    # Return the stress vertices as output
    return verts, MV, theta_bar

# Example FCC and BCC crystal tau array
# tau = np.double(np.array([ \
#     # 110 slip
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     # 112 slip
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1], \
#     [1, 1] \
#     ]))
            
# Example HCP crystal tau array
# tau = np.double(np.array([ \
    # # Basal slip
    # [1, 1], \
    # [1, 1], \
    # [1, 1], \
    # # Prismatic slip
    # [1.25, 1.25], \
    # [1.25, 1.25], \
    # [1.25, 1.25], \
    # # Pyramidal c+a slip
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
    # [1.7, 1.7], \
#     # Pyramidal a slip
#     # [1, 1], \
#     # [1, 1], \
#     # [1, 1], \
#     # [1, 1], \
#     # [1, 1], \
#     # [1, 1] \
    # ]))

# Run code
# verts, MV, theta_bar = scys_calc_scys_vertices('hcp', tau, covera=1.587)