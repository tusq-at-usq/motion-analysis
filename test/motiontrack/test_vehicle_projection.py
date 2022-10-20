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
from motiontrack.sample_bodies.vehicle import make_vehicle
from motiontrack.plot import PlotMatch

def show_camera_viewpoints():

    body = make_vehicle()
    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    view_t = View(body,np.array([-np.pi/2,0.0,0.0]),"top")
    view_w = View(body,np.array([np.pi, 0, np.pi/2]),"west")
    view_e = View(body,np.array([0, 0, np.pi/2]),"east")
    view_f = View(body,np.array([np.pi/2, 0, np.pi/2]),"front")
    views = [view_t, view_w, view_e, view_f]

    plot_t = PlotMatch('Top')
    plot_w = PlotMatch('West')
    plot_e = PlotMatch('East')
    plot_f = PlotMatch('Front')
    plots = [plot_t, plot_w, plot_e, plot_f]
 
    for plot, view in zip(plots, views):
        #  plot.update_projection(view.get_blobs())
        plot.update_mesh(*view.get_mesh())

    body.plot()
    input("Press to close")

if __name__=='__main__':

    show_camera_viewpoints()

