# pylint: disable=invalid-name

""" GEOMETRY using STL format

The local level coordinate system is used for geometry and projection, and is defined as:

            /|\ 1 (north/forwards)
             |
             |
     <-------o 2 ( west/left)
          3 (upwards/out of earth)

In coordinate system is used for dynamical systems, and is defined as:

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)

TODO: Add conversion between local and dynamics coordinate system

The structure of the BodySTL system is that there is a 'fixed' body, and a sub-body projection


Authors: Andrew Lock
First created: September 2022
"""

from typing import List, Optional, Union
import copy
import numpy as np
from numpy import linalg as la
import stl
import vtkplotlib as vpl
from motiontrack.utils import *

class ArucoMarker:
    def __init__(self, aruco_id: int, points: np.array, surface: int, R: np.array = np.eye(3)):
        # TODO: Add rotational vector for initialisation
        self.aruco_id = aruco_id
        self.points = points
        self.surface = surface
        self.R = R

class BodySTL:
    def __init__(self):
        self.Xb = np.array([0, 0, 0]) # Current body position
        self.Q = np.array([1, 0, 0, 0]) # Current body rotation

        self.dot_mk_r0 = np.empty(0)
        self.dot_mk_r = np.empty(0)

        self.mesh_0 = None
        self.mesh = None

        self.dot_surfaces = np.empty(0)
        self.surface_dots = np.empty(0)
        self.dot_mk_s = np.empty(0)

        self.aruco_markers_0 = [] # list of aruco_marker instances
        self.aruco_markers = []

        self.n_faces = []
        self.normal_mags = np.empty(0)

    def import_file(self, filepath: str, scale: float=1):
        """
        Import STL file

        Parameters
        ----------
        filepath : str
            Path of the STL file to import
        scale : float
            Scale of the STL file (i.e. 1/1000 when STL is in mm).
        """
        self.mesh_0 = stl.mesh.Mesh.from_file(filepath)
        self.mesh_0.data['vectors'] *= scale
        self.mesh_0.data['normals'] *= scale**2
        self.n_faces = self.mesh_0.data.shape[0]
        self.normal_mags = la.norm(self.mesh_0.normals,axis=1)

    def define_manually(self, vectors):
        """ Placeholder for alternate geometry definition"""
        pass

    def define_centroid(self, Xc: Optional[List[float]]=None):
        """
        Define body centre of mass, and move centroid to origin. 

        The centre of mass can be defined manuallly, or calculated based on
        mesh geometry (assuming it is solid)

        Parameters
        ----------
        Xc : list (optional)
            The XYZ coordinates of the centroid. If None, it is automatically
            calculated
        """
        if not Xc:
            _, Xc, _, = self.mesh_0.get_mass_properties()
            dX = -1*Xc
        self.mesh_0.translate(dX)
        self.Xb = np.array([0,0,0])
        #  self.mesh_o = copy.copy(self.mesh)

    def _associate_point_with_surface(self, point):
        scale = np.mean(la.norm(self.mesh_0.normals,axis=1))**0.5
        plane_offsets = -1*np.diag(self.mesh_0.vectors[:,0,:]@(self.mesh_0.normals.T))

        distances = np.abs(self.mesh_0.normals@point + plane_offsets)\
            /np.abs(la.norm(self.mesh_0.normals))
        if np.min(distances) > scale/100:
            print("WARNING: point", point, "is greater than 1% units from surface")
        # Candidate surfaces where dot marker almost lies on plane
        candidates = np.where((distances - np.min(distances))<scale/100)[0]
        # If more than one candidate, find the surface with the closest
        # points to the dot marker
        if len(candidates)>1:
            av_dists = np.mean(np.linalg.norm(point-self.mesh_0.vectors[[candidates]],axis=3),axis=2)[0]
            point_surface = candidates[np.argmin(av_dists)]
        else:
            point_surface = candidates[0]
        return point_surface

    def add_dot_markers(self, coords: np.array, sizes: np.array):
        """
        Add dot marker data to geometry

        Dot markers are added by XYZ coordinates and diameters. Each dot is then
        assocaited with a body surface, which controls its visibility for
        2D projections.

        Parameters
        ----------
        coords : np.array
            A 2-dimensional array of size (nx3) of XYZ dot coordinates,
            relative to the body centroid.
        sizes : np.arrary
            A 1-dimensional array of sizes
        """
        self.dot_mk_r0 = coords.astype(float)
        self.dot_mk_s = sizes.astype(float)

        # Associate each dot with a surface
        dot_surfaces = []
        for i,x in enumerate(self.dot_mk_r0):
            surface = self._associate_point_with_surface(x)
            dot_surfaces.append(surface)

        self.dot_surfaces = np.array(dot_surfaces)
        # Create a reference of the dots (by index) on each surface
        self.surface_dots = [[] for _ in range(self.n_faces)]
        for i, dot_surface in enumerate(self.dot_surfaces):
            self.surface_dots[dot_surface].append(i)
        self.surface_dots = np.array(self.surface_dots, dtype=object)

    def add_aruco_code(self, aruco_id, coords, R):
        # Use first point of aruco coord to associate with surface
        surface = self._associate_point_with_surface(coords[0])
        aruco_mk = ArucoMarker(aruco_id, coords, surface, R)
        self.aruco_markers_0.append(aruco_mk)

    def initialise(self,
                   X0: Union[np.array, List[float]],
                   Q0: Union[np.array, List[float]]):

        self.mesh_0.translate(X0)
        T_BL = quaternion_to_rotation_tensor(*Q0)
        self.mesh_0.rotate_using_matrix(T_BL.T, point=self.Xb)

        self.mesh = copy.deepcopy(self.mesh_0)
        self.points = self.points0.copy()
        self.dot_mk_r = copy.deepcopy(self.dot_mk_r0)
        self.aruco_markers = copy.deepcopy(self.aruco_markers_0)
        self.aruco_dict = {a.aruco_id: a for a in self.aruco_markers}
        self.aruco_surface_dict = {i:[] for i in range(self.n_faces)}
        for a in self.aruco_markers:
            self.aruco_surface_dict[a.surface].append(a)
        self.origin_vecs = np.array([[[0.01, 0, 0],
                                      [0, 0, 0]],
                                     [[0, 0.01, 0],
                                      [0, 0, 0]],
                                     [[0, 0, 0.01],
                                      [0, 0, 0]]
                                     ])
        self.body_vecs = self.origin_vecs.copy() 


    def update(self, Xb: np.array, Qb: np.array):
        """
        Update the position and orientation of the body

        Rotation is assumed about the centoid

        Parameters
        ----------
        Xb : np.array
            XYZ coordinates of the new centroid
        Qb : np.array
            The quatnernions of the new orientation (relative to original 
            orientation), with the scalar component as element 0. 
        """
        self.Xb = Xb
        T_L = quaternion_to_rotation_tensor(*Qb)
        self.vectors[:,:,:] = self.to_ndarray(T_L@self.to_vectors(self.vectors0)) + Xb.reshape(-1,1).T
        self.points  = (T_L@self.points0.T).T + Xb

        self.normals[:,:] = (T_L@self.normals0.T).T
        self.dot_mk_r[:,:] = (T_L@self.to_vectors(self.dot_mk_r0)).T.reshape(-1,3) + Xb.reshape(-1,1).T
        # TODO: This may only pass a reference and not update - check
        for marker, marker_0 in zip(self.aruco_markers, self.aruco_markers_0):
            marker.points[:,:] = (T_L@self.to_vectors(marker_0.points)).T.reshape(-1,3) + Xb.reshape(-1,1).T

        for i in range(3):
            self.body_vecs[i,:,:] = (T_L@self.origin_vecs[i].T + Xb.reshape(-1,1)).T


    #  def project_blobs(self, Xb: np.array, Q: np.array):
        #  T_L = quaternion_to_rotation_tensor(*Q)
        #  blobs = self.blob_x.copy()
        #  blobs = (T_L@self.to_vectors(blobs)).T.reshape(-1,3) + Xb.reshape(-1,1).T
        #  return blobs

    def plot(self, block=False):
        #TODO: This could be cleaned up, to remove attribute checking
        if not hasattr(self, 'fig'):
            #  self.fig = vpl.QtFigure2(name='Body mesh plot')
            self.fig = vpl.figure(name='Body mesh plot')
            self.fig.camera.SetParallelProjection(1)
            self.q0 = np.array([1, 0, 0, 0])
            self.view_dict = vpl.view(camera_direction=[0,0,-1], up_view=[1, 0, 0])
            self.q0 = self.get_camera_angle()
            #  vpl.gcf().update()
            #  vpl.reset_camera(fig=self.fig)
        if hasattr(self, 'mesh_plot'):
            self.fig -= self.mesh_plot
        if hasattr(self, 'dot_plot'):
            self.fig -= self.dot_plot
        if hasattr(self, 'aruco_plot'):
            self.fig -= self.aruco_plot
        self.dot_plot = vpl.scatter(self.dot_mk_r, color='k', fig=self.fig, radius=self.dot_mk_s)
        self.aruco_plot = vpl.plot(self.aruco_points, color='r', fig=self.fig)
        self.mesh_plot = vpl.mesh_plot(self, fig=self.fig, opacity=1)

        vpl.text3d('top',np.array([0, 0, 0.0146]), scale=1e-3, color='k', fig=self.fig)
        vpl.text3d('front',np.array([0.0146, 0, 0.0]), scale=1e-3, color='k', fig=self.fig)
        vpl.text3d('back',np.array([-0.0146, 0, 0]), scale=1e-3, color='k', fig = self.fig)
        vpl.text3d('bottom',np.array([0, 0, -0.0146]), scale=1e-3, color='k', fig = self.fig)
        vpl.text3d('left',np.array([0, 0.0146, 0.0]), scale=1e-3, color='k', fig=self.fig)
        vpl.text3d('right',np.array([0, -0.0146, 0.]), scale=1e-3, color='k', fig = self.fig)
        
        #  self.mesh_plot = vpl.plot(self, fig=self.fig, opacity=1)
        #  vpl.reset_camera(fig=self.fig)
        vpl.gcf().update()
        vpl.zoom_to_contents()
        self.fig.show(block=block)

    def get_camera_angle(self):
        #  q0 = (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)
        self.view_dict = vpl.view()
        wxyz = self.fig.camera.GetOrientationWXYZ()
        w_rad = np.deg2rad(wxyz[0])
        q_total = wxyz_to_quaternion(w_rad, wxyz[1], wxyz[2], wxyz[3])
        q_rot = quaternion_subtract(q_total, self.q0)
        self.q0 = q_total
        self.view_dict = vpl.view(camera_direction=[0,0,-1], up_view=[1, 0, 0])
        vpl.reset_camera(fig=self.fig)
        vpl.gcf().update()
        return q_rot

    def get_face_centroids(self):
        centroids = np.average(self.vectors,axis=1)
        return centroids

    @property
    def face_centres(self):
        return self.get_face_centroids()

    @property
    def normals(self):
        return self.mesh.data['normals']

    @property
    def vectors(self):
        return self.mesh.data['vectors']

    @property
    def dots(self):
        return self.dot_mk_r

    @property
    def arucos(self):
        return self.aruco_markers

    @property
    def aruco_id_dict(self):
        return self.aruco_dict

    @property
    def aruco_points(self):
        x = np.concatenate([a.points for a in self.aruco_markers])
        return x

    @property
    def normals0(self):
        return self.mesh_0.data['normals']

    @property
    def vectors0(self):
        return self.mesh_0.data['vectors']

    @property
    def points0(self):
        return  np.unique(self.mesh_0.points.reshape(-1,3), axis=0)

    def to_vectors(self, x):
        return x.reshape(-1, 3).T

    def to_ndarray(self, x):
        return x.T.reshape(self.n_faces, 3, 3)

    def to_2d_ndarray(self, x):
        return x.T.reshape(self.n_faces, 3, 2)

    @property
    def unit_normals(self):
        return (self.normals.T/self.normal_mags).T

    @property
    def dot_surfs(self):
        return self.dot_surfaces

    @property
    def body_points(self):
        return self.points

    @property
    def dot_sizes(self):
        return self.dot_mk_s

    @property
    def surf_dots(self):
        return self.surface_dots

    @property
    def unit_vecs(self):
        return self.body_vecs

    def to_mesh(self, x):
        return self.to_ndarray(x)

    def to_2d_mesh(self, x):
        return self.to_2d_ndarray(x)





