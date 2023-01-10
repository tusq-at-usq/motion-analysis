""" Example geometry projection
This code shows an example of how the geometry is projected by the viewpoint.
Care must be taken with the Euler roation of camera view angle (easy to get wrong).
The object is rotated about the camera BODY (not local) coordinate axis.
The order of rotation (positive RHR direction: Z, Y, X body axis).

Below is an examaple of the camera set up for different viewpoints,
as well as a brief animation showing three consecutive rotations.

"""

import numpy as np
from motiontrack.body_projection import CameraView
from motiontrack.utils import euler_to_quaternion
from motiontrack.sample_bodies.cube import make_cube

from motiontrack.plot import PlotMatch

def load_stl(trigger_start=False):

    body = make_cube()

    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    steps = 300

    Q2 = np.linspace(0,0.5,50)
    Q3 = np.linspace(0,2,50)
    Q1 = np.linspace(0,np.pi,50)

    X1 = np.linspace(0,50,100)

    view_t = CameraView(body,np.array([-np.pi/2,0.0,0.0]),"top",'test',0)
    view_e = CameraView(body,np.array([0, 0, np.pi/2]),"east",'test',0)
    views = [view_t, view_e]

    plot_t = PlotMatch('Top')
    plot_e = PlotMatch('East')
    plots = [plot_t, plot_e]
    body.plot()
    if trigger_start:
        input("Press key to start animation")

    for q1, q2, q3 in zip(Q1, Q2, Q3):
        Q = euler_to_quaternion(q1, q2, q3)
        body.update([0,0,0], [Q[0], Q[1], Q[2], Q[3]])
        for view, plot in zip(views, plots):
            blob_data = view.get_blobs()
            mesh_data = view.get_mesh()
            plot.update_projection(blob_data)
            plot.update_mesh(*mesh_data)
        body.plot()
    if trigger_start:
        input("Press key to close")

    plot.close()


if __name__=='__main__':

    load_stl()

