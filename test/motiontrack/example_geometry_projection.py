from motiontrack.geometry import *
from motiontrack.vehicle_image_generation import *
from matplotlib import pyplot as plt

from motiontrack.utils import *

""" Example geometry projection
This code shows an example of how the geometry is projected by the viewpoint.
Care must be taken with the Euler roation of camera view angle (easy to get wrong).
The object is rotated about the BODY (not local) coordinate axis.
The order of rotation (positive RHR direction: Z, Y, X body axis).

Below is an examaple of the camera set up for different viewpoints, 
as well as a brief animation showing three consecutive rotations. 

The vehicle/body coordinate system is defined as

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)
"""

def show_camera_viewpoints():

    G = pyramid_gen(1)

    V_b = View(G,np.array([0.0,0.0,np.pi]),"bottom",'test',0)
    V_r = View(G,np.array([0.0,np.pi/2,0.0]),"rear",'test',0)
    V_f = View(G,np.array([np.pi,np.pi/2,0]),"front",'test',0)
    V_t = View(G,np.array([0.0000,0.0000,0.000000]),"top",'test',0)
    V_w = View(G,np.array([np.pi/2,np.pi/2,0.0]),"west",'test',0)
    V_e = View(G,np.array([-np.pi/2,np.pi/2,0.0]),"east",'test',0)
    
    Vs = [V_b,V_r,V_f,V_t,V_w,V_e]
        
    for V in Vs:
        V.update()
        V.plot_vehicle()
    plt.pause(0.1)
    input("Press to close:")
    plt.close('all')

def create_example_view_rotation():
    G = pyramid_gen(1)
    V = View(G,np.array([0.0000,0.00000,0.000000]),"example",'example',1)

    angles = np.linspace(0,1.75*np.pi,100)

    r1 = []
    r2 = []
    r3 = []
    for angle in angles:
        r1.append(euler_to_quaternion(angle,0,0))
        r2.append(euler_to_quaternion(1.75*np.pi,angle,0.0))
        r3.append(euler_to_quaternion(1.75*np.pi,1.75*np.pi,angle))

    q_ar = r1+r2+r3
    titles = ["z rotation"]*100 + ["y rotation"]*100 + ["x rotation"]*100

    for q,title in zip(q_ar,titles):
        G.update([0,0,0],q,0)
        V.update()
        V.plot_vehicle(title)

    
show_camera_viewpoints()
create_example_view_rotation()

