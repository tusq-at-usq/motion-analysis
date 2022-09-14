
import copy
import numpy as np
from numpy import linalg as la
import stl
import vtkplotlib as vpl
from scipy.spatial.transform import Rotation as R


""" GEOMETRY using STL format
Define a body for using in graphics creation/interpretation of results
Geometry is handled by Points, Surfaces (made of points), and Body (made of surfaces).
After initialisation, a local coordinate system X,Y,Z and local->body psi,theta,phi are updated at each time step.
Only surfaces which have a positive dot product of surface normal vector and camera position are plotted.

The local level coordinate system is used for geometry and projection, and is defined as:

    /|\ 1 (north)
     |
     |
     X--------> 2 (east)
  3 (into earth)

The vehicle/body coordinate system is used for dynamical systems, and is defined as:

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)

Authors: Andrew Lock
First created: September 2022
"""

def calculate_direction_cosine_from_quaternions(q0, q1, q2, q3):
    T_rot = R.from_quat([q1, q2, q3, q0]).as_matrix()
    return T_rot

class BodyProjection:
    """ An additional mesh object which defines rotation and translation
    from an original position (instead of relative as in numpy-stl)
    """
    def __init__(self,
                 mesh0,
                 blob_x0,
                 blob_surfaces):
        self.mesh0 = mesh0
        self.mesh = mesh0.copy()
        self.blob_x0 = blob_x0
        self.blob_x = blob_x0.copy()
        self.blob_surfaces = blob_surfaces
        self.n_faces = self.mesh0.shape[0]

    @property
    def normals(self):
        return self.mesh['normals']

    @property
    def vectors(self):
        return self.mesh['vectors']

    @property
    def blobs(self):
        return self.blob_x

    @property
    def normals0(self):
        return self.mesh0['normals']

    @property
    def vectors0(self):
        return self.mesh0['vectors']

    def to_vectors(self, x):
        return x.reshape(-1, 3).T

    def to_ndarray(self, x):
        return x.T.reshape(self.n_faces, 3, 3)

    def update(self, Xb, Qb):
        T_L = calculate_direction_cosine_from_quaternions(*Qb)
        self.vectors[:,:,:] = self.to_ndarray(T_L@self.to_vectors(self.vectors0))
        self.vectors[:,:,0] += Xb[0]
        self.vectors[:,:,1] -= Xb[1]
        self.vectors[:,:,2] -= Xb[2]
        self.normals[:,:] = (T_L@self.normals0.T).T
        self.blob_x[:,:] = (T_L@self.to_vectors(self.blob_x0)).T.reshape(-1,3)
        self.blob_x[:,0] += Xb[0]
        self.blob_x[:,1] -= Xb[1]
        self.blob_x[:,2] -= Xb[2]

    # TODO: Initialise plot properly
    def plot(self):
        if not hasattr(self, 'fig'):
            self.fig = vpl.QtFigure(name='Body mesh plot')
            vpl.view(camera_direction=[0,1,0], up_view=[0, 0, 1])
            vpl.gcf().update()
            vpl.reset_camera(fig=self.fig)
        if hasattr(self, 'mesh_plot'):
            self.fig -= self.mesh_plot
        if hasattr(self, 'blob_plot'):
            self.fig -= self.blob_plot
        self.blob_plot = vpl.scatter(self.blob_x, color='k', fig=self.fig)
        self.mesh_plot = vpl.mesh_plot(self, fig=self.fig, opacity=1)
        vpl.gcf().update()
        vpl.reset_camera(fig=self.fig)
        self.fig.show(block=False)

class BodySTL:
    def __init__(self):
        self.Xb = np.array([0, 0, 0]) # Current body position
        self.Q = np.array([1, 0, 0, 0]) # Current body rotation

        self.blob_surfaces = np.empty(0)
        self.surface_blobs = np.empty(0)
        self.blob_s = np.empty(0)

        self.n_faces = []
        self.normal_mags = np.empty(0)

    def import_file(self, filename):
        self.mesh = stl.mesh.Mesh.from_file(filename)
        self.n_faces = self.mesh.data.shape[0]
        #  if units == 'mm':
            #  self.mesh.data['vectors'] *= 1/1000
            #  self.mesh.data['normals'] *= 1/1000
        self.normal_mags = la.norm(self.mesh.normals,axis=1)
    def define_manually(self, vectors):
        pass

    def define_centroid(self, Xc=None):
        if not Xc:
            _, cog, _, = self.mesh.get_mass_properties()
            Xc = -1*cog
        self.mesh.translate(Xc)
        self.Xb = np.array([0,0,0])
        self.mesh_o = copy.copy(self.mesh)

    def add_blobs(self, coords, sizes):
        # blobs are currently defined from the body cog
        self.blob_x = coords.astype(float)
        self.blob_s = sizes
        blob_surfaces = []

        scale = np.mean(la.norm(self.mesh.normals,axis=1))**0.5
        plane_offsets = -1*np.diag(self.mesh.vectors[:,0,:]@(self.mesh.normals.T))

        for i,x in enumerate(self.blob_x):
            distances = np.abs(self.mesh.normals@x + plane_offsets)\
                /np.abs(la.norm(self.mesh.normals))
            if np.min(distances) > scale/100:
                print("WARNING: Blob",i," greater than 1% units from surface")
            candidates = np.where((distances - np.min(distances))<scale/100)[0]
            if len(candidates)>1:
                #  print("Multiple intersecting surfaces found for blob",i)
                av_dists = np.mean(np.linalg.norm(x-self.mesh.vectors[[candidates]],axis=3),axis=2)[0]
                blob_surfaces.append(candidates[np.argmin(av_dists)])
            else:
                blob_surfaces.append(candidates[0])
        self.blob_surfaces = np.array(blob_surfaces)

        # Create a reference of the blobs (by index) on each surface
        self.surface_blobs = [[] for _ in range(self.n_faces)]
        for i,blob_surface in enumerate(self.blob_surfaces):
            self.surface_blobs[blob_surface].append(i)
        self.surface_blobs = np.array(self.surface_blobs, dtype=object)

    def initialise(self, X_offset, Q_offset):
        # TODO: Confirm X and Q offset 
        #  q0,q1,q2,q3 = Q0
        #  T_BL = calculate_direction_cosine_from_quaternions(q0,q1,q2,q3)
        #  self.mesh.rotate_using_matrix(T_BL.T, point=self.Xb)
        #  self.mesh.translate(X0)
        #  self.blob_x =(T_BL@(self.blob_x-self.Xb).T).T+self.Xb
        #  self.Xb = self.Xb + X0
        self.sub_body = BodyProjection(self.mesh.data,
                                   self.blob_x,
                                   self.blob_surfaces)
    def update(self, Xb, Qb):
        self.sub_body.update(Xb, Qb)

    @property
    def normals(self):
        return self.sub_body.normals

    @property
    def unit_normals(self):
        return (self.sub_body.normals.T/self.normal_mags).T

    @property
    def blobs(self):
        return self.sub_body.blobs

    @property
    def blob_surfs(self):
        return self.blob_surfaces

    @property
    def blob_sizes(self):
        return self.blob_s

    @property
    def surf_blobs(self):
        return self.surface_blobs

    @property
    def vectors(self):
        return self.sub_body.vectors

    def to_vectors(self, x):
        return self.sub_body.to_vectors(x)

    def to_mesh(self, x):
        return self.sub_body.to_ndarray(x)

    def plot(self):
        self.sub_body.plot()



