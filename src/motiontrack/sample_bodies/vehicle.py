""" Sample STL body with surface blobs
"""

import os
import numpy as np
from motiontrack.geometry import BodySTL

STL_PATH = os.path.join(os.path.dirname(__file__), 'data/vehicle.stl')

def make_vehicle():
    body = BodySTL()
    body.import_file(STL_PATH)
    body.define_centroid()
    blob_x = np.array([])
    sizes = np.full(len(blob_x),0.01)
    body.add_blobs(blob_x, sizes)

    return body

if __name__=='__main__':
    from motiontrack.utils import euler_to_quaternion
    body = make_vehicle()
    Q = euler_to_quaternion(0, -np.pi, -np.pi/2)
    body.initialise([0,0,0], Q)
    body.plot()
    input("Press key to close")


