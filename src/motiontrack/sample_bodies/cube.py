""" Sample STL body with surface blobs
"""

import numpy as np
from motiontrack.geometry_stl import BodySTL

STL_NAME = 'data/cube.stl'

def make_cube():
    body = BodySTL()
    body.import_file(STL_NAME)
    body.define_centroid()
    blob_x = np.array([
        [-50, 0, 0], # Back
        [50, 0, 15], # Front
        [50, 0, -15], # Front
        [-20, 50, -25], # Left
        [20, 50, -25], # Left
        [0, 50, 25], # Left
        [-20, -50, 20], # Right
        [20, -50, 20], # Right
        [-20, -50, -20], # Right
        [20, -50, -20], # Right
        [45, 45, 50], # Top
        [-45, -45, 50], # Top
        [45, -35, -50], # Bottom
        [-45, -45, -50], # Bottom
        [45, 45, -50], # Bottom
        [-45, 45, -50] # Bottom
    ])
    sizes = np.full(len(blob_x),5)
    body.add_blobs(blob_x, sizes)
    return body


