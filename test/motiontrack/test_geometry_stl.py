""" Example geometry projection
This code shows an example of how the geometry is projected by the viewpoint.
Care must be taken with the Euler roation of camera view angle (easy to get wrong).
The object is rotated about the camera BODY (not local) coordinate axis.
The order of rotation (positive RHR direction: Z, Y, X body axis).

Below is an examaple of the camera set up for different viewpoints,
as well as a brief animation showing three consecutive rotations.

"""

import os
import numpy as np
from motiontrack.camera import CameraCalibration, CameraView
from motiontrack.utils import euler_to_quaternion, euler_to_rotation_tensor
from motiontrack.plot import PlotMatch
from motiontrack.geometry import BodySTL

STL_PATH = os.path.dirname(__file__)+'/data/cube.stl'

PLOT_ANIMATION = False

def make_cube():
    body = BodySTL()
    body.import_file(STL_PATH, scale=1e-3)
    body.define_centroid()
    blob_x = np.array([
        [-50, 0, 0], # Back
        [-50, -30, 0], # Back
        [-50, 10, 25], # Back
        [50, 0, 15], # Front
        [50, 40, -15], # Front
        [50, 25, -25], # Front
        [50, -39, -4], # Front
        [-20, 50, -25], # Left
        [20, 50, -25], # Left
        [0, 50, 25], # Left
        [6, 50, -12], # Left
        [-12, -50, 20], # Right
        [15, -50, 38], # Right
        [-38, -50, -20], # Right
        [41, -50, -17], # Right
        [20, -50, -20], # Right
        [45, 45, 50], # Top
        [-45, -45, 50], # Top
        [21, 15, 50], # Top
        [45, -35, -50], # Bottom
        [-45, -45, -50], # Bottom
        [45, 45, -50], # Bottom
        [-45, 45, -50] # Bottom
    ])*1e-3
    sizes = np.full(len(blob_x),0.002)
    body.add_blobs(blob_x, sizes)
    return body

def test_projection():

    body = make_cube()

    Q = euler_to_quaternion(0, 0, 0)
    X = np.array([0.0, 0.0, 0.0])
    body.initialise(X, Q)
    X = np.array([0, 0, 0])

    steps = 300

    E1 = np.linspace(0,np.pi,50)
    E2 = np.linspace(0,0.5,50)
    E3 = np.linspace(0,2,50)

    X1 = np.linspace(0,50,100)

    dist = np.zeros(14)
    mtx = np.array([[512, 0, 512],
                    [0, 512, 512],
                    [0, 0, 1]])

    R = euler_to_rotation_tensor(0, -np.pi/2, 0)
    R_L = np.array([[0., -1., 0.],
                  [-1., 0., 0.],
                  [0., 0., -1.]])
    T = np.array([0, 0, 1])


    cal1 = CameraCalibration(mtx, dist, T=T, R_L=R_L)
    cal2 = CameraCalibration(mtx, dist, R, T=T, R_L=R_L)

    view_t = CameraView(body, cal1)
    view_e = CameraView(body, cal2)
    views = [view_t, view_e]


    if PLOT_ANIMATION:
        plot_t = PlotMatch('Top')
        plot_e = PlotMatch('East')
        plots = [plot_t, plot_e]
        body.plot()
        for e1, e2, e3 in zip(E1, E2, E3):
            Q = euler_to_quaternion(e1, e2, e3)
            body.update(X, Q)
            for view, plot in zip(views, plots):
                blob_data = view.get_blobs()
                mesh_data = view.get_mesh()
                plot.update_projection(blob_data)
                plot.update_mesh(*mesh_data)
            body.plot()
        plot_t.close()
        plot_3.close()

    Q = euler_to_quaternion(E1[-1], E2[-1], E3[-1])
    body.update(X, Q)
    blob_data_t = view_t.get_blobs().points
    blob_data_e = view_e.get_blobs().points

    blob_data_t_true = np.array([506.71451869, 527.73324215])
    blob_data_e_true = np.array([533.41127362, 527.23254406])

    np.testing.assert_almost_equal(blob_data_t[-1], blob_data_t_true)
    np.testing.assert_almost_equal(blob_data_e[-1], blob_data_e_true)

    return body

if __name__=='__main__':

    body = test_projection()
    body.plot()
    input("Press to continue")

