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

from motiontrack.spatial_match import *
from motiontrack.geometry import *
from motiontrack.vehicle_image_generation import *
from motiontrack.utils import *
from motiontrack.read_data import BlobsFrame

def test_spatial_match(print_flag=0):

    # Create cube body with pre-determined blob locations
    B = cube_gen(1,'data/blob_XYs.npy')

    V_t = View(B,np.array([0.,0.,0.]),'top') # View from top
    V_w = View(B,np.array([np.pi/2,np.pi/2,0.]),'west') # View from west

    p = np.random.rand(3)*5 # Random XYZ
    EA = np.full(3,-np.pi) +  np.random.rand(3)*2*np.pi # Random Euler angles
    Q = euler_to_quaternion(*EA) 

    B.update(p,Q)
    V_t.update()
    V_w.update()

    # True blob locations and diameters
    blobs_t = V_t.get_2D_data()
    blobs_w = V_w.get_2D_data()


    # Create initial estimated guess with error
    p_est = p + np.random.rand(3)*0.1
    EA_est = EA - np.full(3,np.pi/16) + np.random.rand(3)*np.pi/8

    Q_est = euler_to_quaternion(*EA_est)
    Q_est = Q_est / np.linalg.norm(Q_est) # Normalise quaternions

    # Add gaussian error to X and Y blob data (to approximate imperfect measured data)
    blobs_er_t = BlobsFrame(blobs_t.points + np.random.normal(scale=0.005,size=blobs_t.n),blobs_t.diameters)
    blobs_er_w = BlobsFrame(blobs_w.points + np.random.normal(scale=0.005,size=blobs_w.n),blobs_w.diameters)

    # Remove two random blobs from each view (to simulate imcomplete data)
    n1 = np.random.randint(0,blobs_t.n-1)
    n2 = np.random.randint(0,blobs_t.n-1)
    blobs_er_t.remove_blob(n1)
    blobs_er_t.remove_blob(n2)

    n3 = np.random.randint(0,blobs_w.n-1)
    n4 = np.random.randint(0,blobs_w.n-1)
    blobs_er_w.remove_blob(n3)
    blobs_er_w.remove_blob(n4)

    # Initialise spatial match object
    S = SpatialMatch(B,[V_t,V_w])

    # Match position and orientation
    p_, Q_ = S.run_match([blobs_er_t,blobs_er_w],p_est,Q_est,plot=1)
    plt.close()

    if print_flag:
        print("----- TEST COMPLETE -----")
        print("True state:")
        print("p = ",p)
        print("E = ",quaternion_to_euler(*Q))
        print("Initialised state:")
        print("p = ",p_est)
        print("E = ",quaternion_to_euler(*Q_est))
        print("Matched state:")
        print("p = ",p_)
        print("E = ",quaternion_to_euler(*Q_))



if __name__=='__main__':
    test_spatial_match(1)



