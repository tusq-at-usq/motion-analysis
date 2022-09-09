""" Example geometry projection
This code shows an example of how the geometry is projected by the viewpoint.
Care must be taken with the Euler roation of camera view angle (easy to get wrong).
The object is rotated about the BODY (not local) coordinate axis.
The order of rotation (positive RHR direction: Z, Y, X body axis).

Below is an examaple of the camera set up for different viewpoints,
as well as a brief animation showing three consecutive rotations.

Geometry and camera views are handled in typical coordinate system:

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)
"""

import numpy as np
from motiontrack.body_projection import View
from motiontrack.utils import euler_to_quaternion
from motiontrack.sample_bodies.cube import make_cube
from motiontrack.plot import PlotMatch

def show_camera_viewpoints():

    body = make_cube()
    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    view__t = View(body,np.array([-np.pi/2,0.0,0.0]),"top",'test',0)
    view__w = View(body,np.array([np.pi, 0, np.pi/2]),"west",'test',0)
    view__e = View(body,np.array([0, 0, np.pi/2]),"east",'test',0)
    view__f = View(body,np.array([np.pi/2, 0, np.pi/2]),"front",'test',0)
    views = [view__t, view__w, view__e, view__f]

    plot_t = PlotMatch('Top')
    plot_w = PlotMatch('West')
    plot_e = PlotMatch('East')
    plot_f = PlotMatch('Front')
    plots = [plot_t, plot_w, plot_e, plot_f]
     
    for plot, view in zip(plots, views):
        view.update_blobs()
        plot.update_projection(view.get_2D_data())

    body.plot()
    input("Press to close")

if __name__=='__main__':

    show_camera_viewpoints()

