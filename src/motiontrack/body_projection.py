#!/usr/bin/python3.8
import sys
import os
from getopt import getopt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pyqtgraph as pg

from motiontrack.utils import *
from motiontrack.blob_data import BlobsFrame

""" View generation


Note that all geometry creation/projection is done in local coordinates
The local level coordinate system is defined as:

          /|\ 1 (Forward/North)
           |
           |
   <-------o 3 (Upwards/out of earth)
2 (Left/West)

Thsi shouldn't be confused with common body coordiante system such as the vehicle/body coordinate system,defined as

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)


Authors: Andrew Lock
First created: Jan 2022

"""

class View:
    # A projection of the 3D local coordinates to a specific 2D direction
    # TODO - can add a perspective projection option later if needed
    def __init__(self,
                 body,
                 view_angle=[0,0,0],
                 name=None,
                 dir_name="default",
                 save_switch=False,
                 scale=1,
                 perspective="parallel",
                 offset = 0):
        self.perspective=perspective
        self.view_angle = view_angle
        self.s_LV = [] # Vector from local origin to camera origin in local coordinates
        self.body = body
        self.cent_point = [] # Body centroid point in camera view
        self.axis_points = [] # Body axis points in camera view
        self.name = name
        self.dir_name = dir_name
        self.artists = []
        self.scale = scale  #Scale of the view image in pixels/m
        self.offset = offset # Placeholder offset
        self.initialisation()

    def initialisation(self):
        # Create a rotation matrix between local frame and camera frame
        q0,q1,q2,q3 = euler_to_quaternion(*self.view_angle)
        self.T_VL = quaternions_to_rotation_tensor(q0,q1,q2,q3).T
        self.s_LV = np.matmul(np.array([0,0,1]),self.T_VL)

    def get_blobs(self, angle_threshold=0.01):
        # Process:
        # 1. Find which surfaces have a positive normal
        # 2. Calculate blob projection onto 2D plane
        # 3. Caluclate blob size
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_blobs = np.array([j for i in visible_surfs \
                                  for j in self.body.surface_blobs[i]])
        visible_blob_x = self.body.blobs[visible_blobs]
        blob_2d = (self.T_VL@visible_blob_x.T).T[:,:2]

        blob_sizes = self.body.blob_sizes[visible_blobs]*\
            dot_prods[self.body.blob_surfaces[visible_blobs]].T[0]

        return BlobsFrame(blob_2d, blob_sizes)

    def get_mesh(self, angle_threshold=0.01):
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_mesh = self.body.to_mesh(
            self.T_VL@self.body.to_vectors(self.body.vectors)
        )[visible_surfs]
        visible_angles = dot_prods[visible_surfs]
        return visible_mesh, visible_angles



