# pylint: disable=invalid-name, too-many-arguments

""" Camera Projection code

Includes:
    - CameraCalibration: Class which stores camera calibration and reprojects
      points in 3D space to 2D coordinates
    - CameraView: Class which takes a Body object and returns 2D projection of
      its mesh, points, Aruco codes, axes etc.

"""

from typing import List, Tuple
import numpy as np
import cv2 as cv

from motiontrack.utils import euler_to_quaternion
from motiontrack.features import BlobsFrame
from motiontrack.geometry import BodySTL

class CameraCalibration:
        """
        Camera Calibration  

        Stores camera calibration data and reprojects points in local 
        3D coordinates to 2D camera coordinates

        Parameters
        ----------
        mtx: np.array
            Camera matrix
        dist: np.array
            Camera distortion coefficients
        R: np.array
            Rotation matrix to primary camera
        T: np.array
            Translation vector to primary camera
        R_L: np.array
            Rotation matrix from primary camera to local frame
        """
    def __init__(self,
                 mtx: np.array,
                 dist: np.array,
                 R: np.array = np.eye(3),
                 T: np.array = np.array([0.,0.,0.]),
                 R_L: np.array = np.eye(3)):
        self.mtx = mtx
        # OpenCV outputs dist as a 1xn 2-dimensional vector by default
        if dist.ndim > 1:
            dist = dist[0]
        self.dist = dist
        self.R = R # Rotation from camera 2 to camera 1
        self.T = T # Translation from camera 2 to camera 1
        self.R_L = R_L # Rotation from C1 openCV to local coordiantes

        self.init()


    def init(self):
        self.dist = np.pad(self.dist,(0,14-self.dist.shape[0]))
        tau_x = self.dist[12]
        tau_y = self.dist[13]
        R_ = np.array([[np.cos(tau_y), np.sin(tau_y)*np.sin(tau_x), -np.sin(tau_y)*np.cos(tau_x)],
                      [0, np.cos(tau_x), np.sin(tau_x)],
                      [np.sin(tau_y), -np.cos(tau_y)*np.sin(tau_x), np.cos(tau_y)*np.cos(tau_x)]])
        self.R_cor = np.array([[R_[2,2], 0, -R_[0,2]],
                               [0, R_[2,2], -R_[1,2]],
                               [0, 0, 1]]) @ R_

    def set_intrinsic(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.init()

    def set_extrinsic(self, R, T):
        self.R = R
        self.T = T
        
    def set_all(self, mtx, dist, R, T):
        self.set_intrinsic(mtx, dist)
        self.set_extrinsic(R, T)

    #  Opencv implementation
    #  def _project(self, X, R_C=None, T=None):
        #  if R_C is None:
            #  R_C = self.R
        #  if T is None:
            #  T = self.T
        #  Y = cv.projectPoints(self.R_L@X, R_C, T, self.mtx, self.dist)
        #  Y = Y[0].reshape(-1,2)
        #  return Y

    def project_wo_distortion(self, x):
        """
        Project points in 3D local coordinates to 2D camera
        image coordinates without distortion correction

        Parameters
        ----------
        x : np.array
            3 x n array of points in 3D coordinates

        Returns
        --------
        Y : np.array
            2 x n array of points in camera image coordiantes
        """
        # Project 3D points without distortion correction
        x_ = self.R_L@x
        P_w = np.vstack((x_,np.full(x_.shape[1],1)))
        RT = np.hstack((self.R,self.T.reshape(-1,1)))
        P_c = RT@P_w
        Z_c = P_c[2]
        xy = P_c/Z_c
        Y = self.mtx@xy
        Y /= Y[2]
        Y = Y[0:2].T.reshape(-1,2)
        return Y

    def _project(self, X, R_C=None, T=None):
        """

        Project points from 3D local coordatines to 2D camera image coordiantes

        Parameters
        ----------
        X : np.arrray
            3 x n array of 3D points in local coordinates 
        R_C : np.array
            Optionally override rotation tensor
        T : np.array
            Optionally override translation vector

        Returns
        -------
        Y : np.array
            2 x n array of points in camera image coordinates
        """

        if R_C is None:
            R_C = self.R
        if T is None:
            T = self.T
        X_ = self.R_L@X
        P_w = np.vstack((X_,np.full(X_.shape[1],1)))
        RT = np.hstack((R_C,T.reshape(-1,1)))
        P_c = RT@P_w
        Z_c = P_c[2]
        xy = P_c/Z_c
        r2 = (xy[0]**2 + xy[1]**2)

        dist = self.dist
        t1 = (1 + dist[0]*r2 + dist[1]*r2**2 + dist[4]*r2**3)/ \
            (1 + dist[5]*r2 + dist[6]*r2**2 + dist[7]*r2**3)
        tx1 = 2*dist[2]*xy[0]*xy[1]
        ty2 = 2*dist[3]*xy[0]*xy[1]
        tx2 = dist[3]*(r2 + 2*xy[0]**2)
        ty1 = dist[2]*(r2 + 2*xy[1]**2)

        xy_cor = np.full(xy.shape,1.)
        xy_cor[0,:] = xy[0]*t1 + tx1 + tx2 + dist[8]*r2 + dist[9]*r2**2
        xy_cor[1,:] = xy[1]*t1 + ty1 + ty2 + dist[10]*r2 + dist[11]*r2**2

        xy_cor2 = self.R_cor@xy_cor
        Y = self.mtx@xy_cor2
        Y /= Y[2]
        Y = Y[0:2].T.reshape(-1,2)
        return Y

    def project(self, X, R_C=None, T = None):
        return self._project(X, R_C, T)

class CameraView:
    """
    2D view of a Body as seen by a camera

    Parameters
    ----------
    body : BodySTL
        Body object to view by Camera
    cal : CameraCalibration
        Calibration object for projecting  3D -> 2D
    R_L : np.array 
        3 x 3 rotation tensor from local frame to principal camera
    name : string
        Viewpoint name
    resolution : List/Tuple
        Resolution of camera view
    inv : bool
        Whether to invert the image horizontally (often needed for Schlieren)
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

        # Vector of camera Z direction in local coordaintes.
        # Used to determine which faces are visble to camera.
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

    def get_visible_dots(self, angle_threshold: float=0.2):
        """
        Returns dots which are visible to the camera

        Parameters
        ----------
        angle_threshold : float 
            Minimum dot-product of normal vectors for dots to be 'visible'
            
        
        Returns
        -------
        visible_dots : np.array
            index of dots visible to camera

        """
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_dots = np.array([j for i in visible_surfs \
                                  for j in self.body.surface_dots[i]])
        return visible_dots

    def get_dots(self, angle_threshold: float=0.2) -> BlobsFrame:
        """
        Get the pixel locations of dots in 2-dimensional XY coordinates

        Process:
        1. Find which surfaces have a normal dot product above the threshold
        2. Calculate dot projection onto 2D plane
        3. Caluclate dot sizes using dot product

        Note: Dot sizing not currently in use. 
        Note: Further development may identify dots by features (size, shading etc.)

        Parameters
        ----------
        angle_threshold : float, default = 0.2
            The threshold of the dot product between view unit vector
            and face normal vector for which dots are hidden.

        Returns
        ----------
        BlobsFrame
            A BlobsFrame instance representing the Blob 2D data from the
            viewpoint
        """
        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_dots = np.array([j for i in visible_surfs \
                                  for j in self.body.surface_dots[i]])

        visible_dots_xyz = self.body.dots[visible_dots].T
        dots_2d = self._transform(visible_dots_xyz).T

        dot_sizes = self.body.dot_sizes[visible_dots]*\
            dot_prods[self.body.dot_surfaces[visible_dots]].T[0]

        return BlobsFrame(dots_2d, dot_sizes)

    def get_mesh(self,
                 angle_threshold: float=0.02,
                 visible_surfs: np.array = None) -> Tuple[np.array, np.array]:
        """
        Get the pixel locations of mesh in  2-dimensional XY coordinates

        Shows mesh elements with a normal vector greater than threshold.

        Parameters
        ----------
        angle_threshold : float, default = 0.01
            The threshold of the dot product between view unit vector
            and face normal vector for which surfaces are hidden.

        Returns
        -------
        visible_mesh: np.array
            A numpy array with mesh data
        visible_angles: np.array
            A 1-dimensional numpy array with values of normal angle magnitude
        """
        if visible_surfs is None:
            dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
            visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_mesh = self.body.to_2d_mesh(
            self._transform(self.body.to_vectors(self.body.vectors))
        )[visible_surfs]
        #  visible_angles = dot_prods[visible_surfs]
        return visible_mesh, visible_surfs

    def get_uncorrected_mesh(self,
                             angle_threshold: float=0,
                             visible_surfs: np.array = None) -> Tuple[np.array, np.array]:
        """
        Get the pixel-locations of body mesh without camera distortion correction

        Similar to get_mesh but without camera distortion

        Parameters
        ----------
        angle_threshold : float, default = 0.01
            The threshold of the dot product between view unit vector
            and face normal vector for which surfaces are hidden.
        visible_surfs : np.array, default = None
            An array of visible surface indexes. If None, calculated by normal vector

        Returns
        -------
        visible_mesh: np.array
            A numpy array with mesh data
        visible_angles: np.array
            A 1-dimensional numpy array with values of normal angle magnitude
        """


        if visible_surfs is None:
            dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
            visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_mesh = self.body.to_2d_mesh(
            self.cal.project_wo_distortion(self.body.to_vectors(self.body.vectors)).T
        )[visible_surfs]
        visible_angles = dot_prods[visible_surfs]
        return visible_mesh, visible_angles

    def get_arucos_by_id(self, ids):
        """
        Return Aruco corner points by Aruco ID

        """
        arucos = [self.body.aruco_id_dict[i] for i in ids]
        aruco_points = [self._transform(a.points.T).T for a in arucos]
        return aruco_points

    def get_aruco_R_by_id(self, ids):
        """ 
        Return aruco rotation matrix for input Aruco IDs
        """
        arucos = [self.body.aruco_id_dict[i] for i in ids]
        aruco_Rs = [a.R for a in arucos]
        return aruco_Rs

    def get_visible_arucos(self, angle_threshold=0.05):
        """
        Return corners of visible Aruco IDs projected to camera image

        Parameters
        ----------
        angle_threshold : float
            Minimum dot project of camera and face normal vector
            for surface to be 'visible'

        Returns
        -------
        corners_2d : np.array
            2 x n array of corner location in camera image coordinates
        """

        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_arucos = []
        for s in visible_surfs:
            visible_arucos += self.body.aruco_surface_dict[s]
        aruco_corners = [a.points for a in visible_arucos]
        aruco_corners = np.array(aruco_corners).reshape(-1,3)
        corners_2d = self._transform(aruco_corners.T).T
        return corners_2d

    def get_visible_aruco_ids(self, angle_threshold=0.05):
        """ 
        Get the ID of visible Aruco markers using face normal vectors
        """

        dot_prods = self.body.unit_normals@self.s_LV.reshape(1,3).T
        visible_surfs = np.where(dot_prods > angle_threshold)[0]
        visible_arucos = []
        for s in visible_surfs:
            visible_arucos += self.body.aruco_surface_dict[s]
        aruco_ids = [a.aruco_id for a in visible_arucos]
        return aruco_ids

    def get_points(self):
        """ 
        Return body points 2D camera coordiantes
        """

        points = self.body.body_points
        points = self._transform(points.T)
        return points

    def get_CoM(self):
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

   
    def get_body_vecs(self):
        """
        Return body vectors in 2D camera image coordiantes
        """

        body_vecs = [self._transform(v.T) for v in self.body.unit_vecs]
        return body_vecs

    def project(self, X, Q, threshold=0.2):
        # TODO: Clean up this code. Separate projection of different features

        self.body.update(X, Q)
        dot_frame = self.get_dots(threshold)
        mesh = self.get_mesh()
        return dot_frame, mesh





