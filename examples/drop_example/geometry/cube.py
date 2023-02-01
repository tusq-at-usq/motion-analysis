""" Sample STL body with surface blobs
"""

import os
import numpy as np
from motiontrack.geometry import BodySTL
from numpy import genfromtxt

STL_PATH = os.path.join(os.path.dirname(__file__), 'cube.stl')
BLOB_PATH = os.path.join(os.path.dirname(__file__), 'cube_photos/blob_xyz.csv')

def make_cube():
    SCALE = 1
    body = BodySTL()
    body.import_file(STL_PATH, scale=(SCALE*3.00/10)*1e-3)
    body.define_centroid()
    blob_xyz = np.genfromtxt(BLOB_PATH, delimiter=',')*SCALE/1e3
    sizes = np.full(len(blob_xyz),0.001)
    body.add_blobs(blob_xyz, sizes)
    return body

if __name__ == "__main__":
    make_cube()

