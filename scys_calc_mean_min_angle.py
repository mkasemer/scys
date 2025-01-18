 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the mean of the angle between each vertex and its nearest neighbor.

INPUT:
verts: An array, shape [num_verts, 5], containing the deviatoric stress vertices 
    that comprise the single crystal yield surface, where num_verts is the
    number of vertices. Each vector is a devatoric stress where the components
    are stored in a 5-vector form.

OUTPUT:
mean_min_ang: The mean of the angle between each vertex and its nearest 
    neighbor.

TO DO:
    - All caught up.

First created on Fri Jan 3 12:52:56 2025

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np
from scys_calc_vec_angle import scys_calc_vec_angle

def scys_calc_mean_min_angle(verts):   

    # Calculate the mean angular distance to the nearest neighbor
    num_verts = np.shape(verts)[1]
    angs = np.zeros([num_verts, num_verts])
    min_ang = np.zeros([num_verts, ])
    for i in range(0, num_verts):
        
        for j in range(0, num_verts):

            # Find the angle between each vertex and every other vertex
            angs[i, j] = scys_calc_vec_angle(verts[:, i], verts[:, j])
        
        # For each vertex, find the (second) minimum angle---the nearest 
        # neighbor (the first minimum will be with itself, 0)
        min_ang[i] = np.partition(angs[i, :], 1)[1]
    
    # Find and return the mean angle to the nearest neighbor for all vertices
    mean_min_ang = np.mean(min_ang)
    return mean_min_ang