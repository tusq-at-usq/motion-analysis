# pylint: disable=invalid-name, too-many-arguments

""" Camera projection of body

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
from typing import List, Tuple
import numpy as np

from motiontrack.utils import euler_to_quaternion, quaternion_to_rotation_tensor
from motiontrack.blob_data import BlobsFrame
from motiontrack.geometry import BodySTL
from motiontrack.custom_types import CameraCalibration

class CameraView:
    """
    2D view of a Body as seen by a camera

    Parameters
    ----------
    body : BodySTL
        Body object to view by Camera
    view_angle : tuple, default=(0,0,0)
        Angle of the viewpoint, using Tait-Bryan angles in z-y-x
        Default view is top view looking forwards.
    name : str, default='unnamed_view'
        Camera name
    origin : tuple, defualt = (0,0,0)
        Origin (zero point) of the local frame in pixel coordinates (XY).
        This should be a common point amongst all cameras.
    scale : float, default=1
        Scale of the view in pixels/m
    resolution : tuple, default=(1024, 1024)
        Resolution of the camera angle in pixels (width, height)
    """

    def __init__(self,
                 body: BodySTL,
                 cal: CameraCalibration,
                 R_L: np.array = np.eye(3),
                 name: str='unnamed_view',
                 resolution: Tuple[float] = (1024, 1024),
                 inv = False):

        self.body = body
        self.name = name
        self.resolution = resolution
        self.cal = cal
        self.R_L = R_L

        # Vector of camera viewpoint
        # The theory of this is to calculate the vector in global coordinates
        # of the negative-z vector in local camera coordiantes.
        # This vector is then used to determine which faces are facing the
        # camera.
        self.s_LV = self.cal.R_L.T@self.cal.R.T@np.array([0,0,-1])


    def _transform(self, x: np.array):
        """
        Internal function which converts 3-dimensional, local frame vectors to
        2-dimensional camera pixel coordiantes

        Parameters
        ----------
        x : np.array, size (3 x n)
            A matrix of 3-dimensional vector coordinates to transform

        Returns
        ----------
        y : np.array, size (2 x n)
            A matrix of 2-dimensional pixel coordinates representing the camera frame
        """
        y = self.cal.project(x).T
        return y

    def get_visible_blobs(self, angle_threshold: float=0.1):
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_blobs = np.array([j for i in visible_surfs \
                                  for j in self.body.surface_blobs[i]])
        return visible_blobs

    def get_blobs(self, angle_threshold: float=0.1) -> BlobsFrame:
        """
        Get the pixel locations of blobs in 2-dimensional XY coordinates

        Process:
        1. Find which surfaces have a normal above thrshold
        2. Calculate blob projection onto 2D plane
        3. Caluclate blob sizes using dot product

        Parameters
        ----------
        angle_threshold : float, default = 0.01
            The threshold of the dot product between view unit vector
            and face normal vector for which blobs are hidden.

        Returns
        ----------
        BlobsFrame
            A BlobsFrame instance representing the Blob 2D data from the
            viewpoint
        """
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_blobs = np.array([j for i in visible_surfs \
                                  for j in self.body.surface_blobs[i]])

        visible_blobs_xyz = self.body.blobs[visible_blobs].T
        blob_2d = self._transform(visible_blobs_xyz).T

        blob_sizes = self.body.blob_sizes[visible_blobs]*\
            dot_prods[self.body.blob_surfaces[visible_blobs]].T[0]

        return BlobsFrame(blob_2d, blob_sizes)

    def get_mesh(self, angle_threshold:float=0.05) -> Tuple[np.array, np.array]:
        """
        Get the pixel locations of mesh in  2-dimensional XY coordinates

        Shows mesh elements with a normal vector greater than threshold.

        Parameters
        ----------
        angle_threshold : float, default = 0.01
            The threshold of the dot product between view unit vector
            and face normal vector for which blobs are hidden.

        Returns
        -------
        visible_mesh: np.array
            A numpy array with mesh data
        visible_angles: np.array
            A 1-dimensional numpy array with values of normal angle magnitude
        """
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_mesh = self.body.to_2d_mesh(
            self._transform(self.body.to_vectors(self.body.vectors))
        )[visible_surfs]
        visible_angles = dot_prods[visible_surfs]
        return visible_mesh, visible_angles

    def get_CoM(self) -> np.array:
        """
        Get the location of the body centre in 2-dimensional XY coordinates.

        Returns
        -------
        cent_coords : np.array
            2-dimensional cooridinates of the centre of mass
        """
        centroid = np.array([self.body.Xb.copy()]).T
        cent_coords = self._transform(centroid)
        return cent_coords

    def project(self, X, Q):
        self.body.update(X, Q)
        blob_frame = self.get_blobs()
        mesh = self.get_mesh()
        return blob_frame, mesh





