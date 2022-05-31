# Vehicle tracker codes

A framework for tracking vehicle motion using measured data (initially blob point tracking). 

The structure of the code is shown below: 
![alt text](info/tracker_screenshot.png?raw=true "tracker_outline")
(*This image to be updated/improved*)

Each block shown is a separate module, and is described in further detail below.

More info to come...

## Dependencies 
FilterPy: https://github.com/rlabbe/filterpy


TBC.

## Dynamic system

Provides the estimated (nonlinear) continuous-time dynamic equations of the system, used by the Kalman filter. 

Uses a `sympy` backend of Sysopt:
https://github.com/andrewjlock/sysopt

## Integrator

Integrates the continuous-time dynamic equations to get the model-estimated state. 

$$ \boldsymbol{x'}_k = f( \boldsymbol{x} _{k-1} ) $$

## Kalman filter

Uses an Unscented Kalman filter to track the vehicle motion and combine measurement data with model predictions.

Uses the Kalman filter from `filterpy`:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

A useful resource for understanding Kalman filters and FilterPy is [here](https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

<!-- TODO: Add more detail -->

## Geometry and blob projection

To track vehicle position from blobs, a 3D representation of the vehicle and blob locations is required.
For a given vehicle position and orientation, blob locations can be projected to any 2D view angle.

### Geometry

Geometry is defined by four classes: Point, Surface, Blob,and Body.

Note that geomtry directions are defined as:
X = north
Y = east
Z = down (into earth)

These directions apply to local and body coordiantes.

#### `Point()`
All points are fixed to the rigid body, and are translated and rotated with the body

$$ \boldsymbol{p} ^\mathrm{L} = [\boldsymbol{T}] ^\mathrm{LB} \boldsymbol{p} ^\mathrm{B} + s ^\mathrm{B} _\mathrm{L} $$


where $[\boldsymbol{T}] ^\mathrm{LB}$ is the rotation tensor from vehicle body frame to local frame in local coordinates, and $s ^\mathrm{B} _\mathrm{L}$ is the position of the vehicle centre of gravity in local coordiantes. 

#### `Surface()`
A surface is made of 3 or more points, ordered in a counter-clockwise direction for an ourward surface normal vector (right hand rule).
A check is made that the points lie on the same plane. 

#### `Blob()` 
A blob is essentially a point.
It is currently defined differently, so that additional information can be added later if needed (diameter, color etc).

Blobs are currently appended to a surface, which currently determins the blobs visibility (i.e. when surface normal is pointing towards the viewpoint). 
Future development will define blob visibilty for curved surfaces (cylindrical and spherical). 

#### `Body()`
A body is made up of multiple surfaces.
A warning is displayed if the surfaces do not form a closed object.
The body includes an `update(X,Q)` routine, which updates the local position of all points. 

### Views
A view object is a 2D projection of the vehicle blobs.
Views are defined by their Euler rotation, with rotation vector $[0,0,0]$ looking down on the model towards the local coordinate origin facing north.
In this model Euler angles are rotations about the vehicle z, y and x axis (in that order).
Mutliple different views of the same body can be defined. 

Refer to the example `example_geometry_projection.py` for common view directions and an example of view rotation.

Note: Currently only parallel view (i.e.schlieren) is supported.
Future development will support perspective views.

# Spatial matching
Spatial matching is the process of determining a measured position $[x,y,z] ^\mathrm{L}$ and rotation in quaternions $[q_0,q_1,q_2,q_3] ^\mathrm{BL}$.



