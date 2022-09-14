#!/usr/bin/python3.8

"""Example spatial match

Models a 1x1m cube body with 4 blobs per surface (randomly allocated).
Has two views: from above and west.

Simulates imperfect  data with:
- Random Gaussian error for each data point
- 2 blobs missing for each view

Initial guess position and rotaion has errors:
- Random position error between [0,10cm]
- Random rotation errors betweem [0,pi/8]


Author: Andrew Lock
Created: 1/6/22
"""

import numpy as np

from motiontrack.spatial_match import SpatialMatch
from motiontrack.body_projection import View
from motiontrack.utils import euler_to_quaternion, quaternion_to_euler
from motiontrack.blob_data import BlobsFrame

from motiontrack.sample_bodies.cube import make_cube
from motiontrack.plot import PlotMatch

def test_spatial_match(print_flag=0):

    body = make_cube()
    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    V_t = View(body, np.array([np.pi/2,0.,0.]),'top') # View from top
    V_e = View(body,np.array([0, 0, np.pi/2]),"east",'test',0)

    EA = np.full(3,-np.pi) +  np.random.rand(3)*2*np.pi # Random Euler angles

    p = np.array([0,0,0])
    Q = euler_to_quaternion(*EA)

    body.update(p, Q)

    # True blob locations and diameters
    blobs_t = V_t.get_blobs()
    blobs_e = V_e.get_blobs()

    # Create initial estimated guess with error
    p_est = p + np.random.rand(3)*2
    EA_est = EA  + np.random.rand(3)*np.pi/6

    Q_est = euler_to_quaternion(*EA_est)

    BLOB_ERROR_SCALE = 0

    # Add gaussian error to X and Y blob data (to approximate imperfect measured data)
    blobs_er_t = BlobsFrame(blobs_t.points + \
                            np.random.normal(scale=BLOB_ERROR_SCALE,
                                             size=(blobs_t.n,2)),
                            blobs_t.diameters)
    blobs_er_e = BlobsFrame(blobs_e.points + \
                            np.random.normal(scale=BLOB_ERROR_SCALE,
                                             size=(blobs_e.n,2)),
                            blobs_e.diameters)

    # Remove a random blobs from each view (to simulate imcomplete data)
    #  n1 = np.random.randint(0,blobs_t.n-1)
    #  n2 = np.random.randint(0,blobs_e.n-1)
    #  blobs_er_t.remove_blob(n1)
    #  blobs_er_3.remove_blob(n2)

    plot_t = PlotMatch('Top')
    plot_e = PlotMatch('East')

    # Initialise spatial match object
    S = SpatialMatch(body, [V_t,V_e])

    # Match position and orientation
    p_, Q_ = S.run_match([blobs_er_t, blobs_er_e],p_est,Q_est, [plot_t, plot_e])

    if print_flag:
        
        EA_ = np.array(quaternion_to_euler(*Q_))
        print("----- TEST COMPLETE -----")
        print("True state")
        print(" = ",quaternion_to_euler(*Q))
        print("Initialised state:")
        print("p = ",p_est)
        print("E = ",quaternion_to_euler(*Q_est))
        print("Matched state:")
        print("p = ",p_)
        print("E = ",quaternion_to_euler(*Q_))

        input("Press to close figures")
    plot_t.close()
    plot_e.close()


if __name__=='__main__':
    test_spatial_match(1)



