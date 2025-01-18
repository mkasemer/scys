# SCYS

## Overview

SCYS is a set of routines written in Python to calculate the topology of the rate independent single crystal yield surface. SCYS allows for arbitrary crystal type, crystal geometry, slip systems, and slip system strengths---the latter of which may display asymmetry in terms of intrafamily strength differences, interfamily strength differences, and strength differences between positive and negative slip.

## Usage

The primary routine of SCYS is "scys_calc_scys_vertices", which calculates the single crystal yield surface vertices for a given crystal type, slip system strengths, and (if necessary) crystal geometry.

Slip system strengths must be defined relative to the slip systems defined for a given crystal type, as initialized and calculated in "scys_calc_schmid_vecs".
