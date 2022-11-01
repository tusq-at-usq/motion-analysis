""" Sample STL body with surface blobs
"""

import os
import numpy as np
#  from motiontrack.geometry_stl import BodySTL
from motiontrack.geometry import BodySTL

STL_PATH = os.path.join(os.path.dirname(__file__), 'data/cube.stl')

def make_cube():
    body = BodySTL()
    body.import_file(STL_PATH, scale=1e-3)
    body.define_centroid()
    blob_x = np.array([
        [-50, 0, 0], # Back
        [-50, -30, 0], # Back
        [-50, 10, 25], # Back
        [50, 0, 15], # Front
        [50, 40, -15], # Front
        [50, 25, -25], # Front
        [50, -39, -4], # Front
        [-20, 50, -25], # Left
        [20, 50, -25], # Left
        [0, 50, 25], # Left
        [6, 50, -12], # Left
        [-12, -50, 20], # Right
        [15, -50, 38], # Right
        [-38, -50, -20], # Right
        [41, -50, -17], # Right
        [20, -50, -20], # Right
        [45, 45, 50], # Top
        [-45, -45, 50], # Top
        [21, 15, 50], # Top
        [45, -35, -50], # Bottom
        [-45, -45, -50], # Bottom
        [45, 45, -50], # Bottom
        [-45, 45, -50] # Bottom
    ])*1e-3
    sizes = np.full(len(blob_x),0.002)
    body.add_blobs(blob_x, sizes)

    return body


