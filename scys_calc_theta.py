 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finds the angle between two vectors of arbitrary dimension, ndim.

INPUT:
vec1: A vector, shape [ndim, ].
vec2: A vector, shape [ndim, ].

OUTPUT:
ang: The angle between vec1 and vec2, in degrees.

TO DO:
    - All caught up.

First created on Fri Jan 3 14:07:23 2025

@author: Matthew Kasemer (GitHub: mkasemer)
"""

# Import statements
import numpy as np
import numpy.linalg as la

def scys_calc_theta(vec1, vec2):   

    # Find the dot product
    dots = np.dot(vec1, vec2)

    # Find the product of the norms
    prodn = la.norm(vec1) * la.norm(vec2)

    # Find the angle, clipping to avoid edge cases marginally outside of bounds
    theta = np.degrees(np.arccos(np.clip((dots / prodn), -1, 1)))
    return theta