 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the mean of the angle between each vertex and its nearest neighbor.

INPUT:
verts: An array, shape [MV, 5], containing the deviatoric stress vertices 
    that comprise the single crystal yield surface, where MV is the number of 
    vertices. Each vector is a devatoric stress where the components are stored 
    in a 5-vector form.

OUTPUT:
theta_bar: The mean of the angle between each vertex and its nearest neighbor.

TO DO:
    - All caught up.

First created on Fri Jan 3 12:52:56 2025

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np
from scys_calc_theta import scys_calc_theta

def scys_calc_theta_bar(verts):   

    # Calculate the mean angular distance to the nearest neighbor
    MV = np.shape(verts)[1]
    theta = np.zeros([MV, MV])
    min_theta = np.zeros([MV, ])
    for i in range(0, MV):
        
        for j in range(0, MV):

            # Find the angle between each vertex and every other vertex
            theta[i, j] = scys_calc_theta(verts[:, i], verts[:, j])
        
        # For each vertex, find the (second) minimum angle---the nearest 
        # neighbor (the first minimum will be with itself, 0)
        min_theta[i] = np.partition(theta[i, :], 1)[1]
    
    # Find and return the mean angle to the nearest neighbor for all vertices
    theta_bar = np.mean(min_theta)
    return theta_bar