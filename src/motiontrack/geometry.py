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

class BodySTL:
    def __init__(self):
        self.Xb = np.array([0, 0, 0]) # Current body position
        self.Q = np.array([1, 0, 0, 0]) # Current body rotation

        self.blob_x0 = np.empty(0)
        self.blob_x = np.empty(0)

        self.mesh_0 = None
        self.mesh = None

        self.blob_surfaces = np.empty(0)
        self.surface_blobs = np.empty(0)
        self.blob_s = np.empty(0)

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

    def add_blobs(self, coords: np.array, sizes: np.array):
        """
        Add blob data to geometry

        Blobs are added by XYZ coordinates and diameters. Each blob is then
        assocaited with a body surface, which controls its visibility for
        2D projections.

        Parameters
        ----------
        coords : np.array
            A 2-dimensional array of size (nx3) of XYZ blob coordinates,
            relative to the body centroid.
        sizes : np.arrary
            A 1-dimensional array of sizes
        """
        self.blob_x0 = coords.astype(float)
        self.blob_s = sizes.astype(float)

        # Associate each blob with a surface
        blob_surfaces = []
        scale = np.mean(la.norm(self.mesh_0.normals,axis=1))**0.5
        # Plane offset from origin
        plane_offsets = -1*np.diag(self.mesh_0.vectors[:,0,:]@(self.mesh_0.normals.T))
        for i,x in enumerate(self.blob_x0):
            # Distances from blob to closest point on surface planes
            distances = np.abs(self.mesh_0.normals@x + plane_offsets)\
                /np.abs(la.norm(self.mesh_0.normals))
            if np.min(distances) > scale/100:
                print("WARNING: Blob",i," greater than 1% units from surface")
            # Candidate surfaces where blob almost lies on plane
            candidates = np.where((distances - np.min(distances))<scale/100)[0]
            # If more than one candidate, find the surface with the closest
            # points to the blob
            if len(candidates)>1:
                av_dists = np.mean(np.linalg.norm(x-self.mesh_0.vectors[[candidates]],axis=3),axis=2)[0]
                blob_surfaces.append(candidates[np.argmin(av_dists)])
            else:
                blob_surfaces.append(candidates[0])
        self.blob_surfaces = np.array(blob_surfaces)
        # Create a reference of the blobs (by index) on each surface
        self.surface_blobs = [[] for _ in range(self.n_faces)]
        for i,blob_surface in enumerate(self.blob_surfaces):
            self.surface_blobs[blob_surface].append(i)
        self.surface_blobs = np.array(self.surface_blobs, dtype=object)

    def initialise(self,
                   X0: Union[np.array, List[float]],
                   Q0: Union[np.array, List[float]]):

        self.mesh_0.translate(X0)
        T_BL = quaternion_to_rotation_tensor(*Q0)
        self.mesh_0.rotate_using_matrix(T_BL.T, point=self.Xb)

        self.mesh = copy.deepcopy(self.mesh_0)
        self.blob_x = copy.deepcopy(self.blob_x0)

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
        self.normals[:,:] = (T_L@self.normals0.T).T
        self.blob_x[:,:] = (T_L@self.to_vectors(self.blob_x0)).T.reshape(-1,3) + Xb.reshape(-1,1).T

    #  def project_blobs(self, Xb: np.array, Q: np.array):
        #  T_L = quaternion_to_rotation_tensor(*Q)
        #  blobs = self.blob_x.copy()
        #  blobs = (T_L@self.to_vectors(blobs)).T.reshape(-1,3) + Xb.reshape(-1,1).T
        #  return blobs

    def plot(self):
        #TODO: This could be cleaned up, to remove attribute checking
        if not hasattr(self, 'fig'):
            self.fig = vpl.QtFigure(name='Body mesh plot')
            self.view_dict = vpl.view(camera_direction=[0,1,0], up_view=[0, 0, 1])
            self.fig.camera.SetParallelProjection(1)
            vpl.gcf().update()
            vpl.reset_camera(fig=self.fig)
        if hasattr(self, 'mesh_plot'):
            self.fig -= self.mesh_plot
        if hasattr(self, 'blob_plot'):
            self.fig -= self.blob_plot
        self.blob_plot = vpl.scatter(self.blob_x, color='k', fig=self.fig, radius=self.blob_s)
        self.mesh_plot = vpl.mesh_plot(self, fig=self.fig, opacity=1)
        vpl.gcf().update()
        vpl.reset_camera(fig=self.fig)
        self.fig.show(block=False)

    def get_camera_angle(self):
        q0 = (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)

        self.view_dict = vpl.view()
        wxyz = self.fig.camera.GetOrientationWXYZ()
        w_rad = np.deg2rad(wxyz[0])
        q_total = wxyz_to_quaternion(w_rad, wxyz[1], wxyz[2], wxyz[3])
        q_rot = quaternion_subtract(q_total, q0)
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
    def blobs(self):
        return self.blob_x

    @property
    def normals0(self):
        return self.mesh_0.data['normals']

    @property
    def vectors0(self):
        return self.mesh_0.data['vectors']

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
    def blob_surfs(self):
        return self.blob_surfaces

    @property
    def blob_sizes(self):
        return self.blob_s

    @property
    def surf_blobs(self):
        return self.surface_blobs

    def to_mesh(self, x):
        return self.to_ndarray(x)

    def to_2d_mesh(self, x):
        return self.to_2d_ndarray(x)





