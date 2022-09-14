# `motiontrack`

`motiontrack` incorporates the general class and functions for tracking 

Below is a description of the main capabilities:

## Observations

The structure of the tracking environment allows for any type of observation of
the form 
$$ \mathbf{y}_i = h_i(\mathbf{x}) $$

An abstract class `ObservationGroup()` provides a default interface between an
observation and the tracking code. This class should be inherited for specific
observation types (such as blob matching, IMUs etc.)

One or more instances of `ObservationGroup` are combined into a single set of
observations $\mathbf{y}$, measurement uncertainty matrix $[R]$, and measurement
function $h(\mathbf{x})$. Measurements can be contradictory (such as two
simultaneous position measurements). The measurement function is primarily
used by the extended Kalman filter to create the observation Jacobian $[H]$.

The `combine_observables` includes the functions required to handle multiple
observation groups, such as querying the next measurement timestep (which may
not be all groups if they are not synchronised at a similar frequency),
combining observations, uncertainties, and measurement functions. 

## Geometry

Body geometry is handled by the `BodySTL()` class, and data can be imported in .STL format using the `import_file(filename)`
method. STL files describe the body as a set of triangular surfaces, using the
local coordinate system (x: forward, y: left, z: upwards).

Surface blobs are added using sets of (x,y,z) coordinates. The surface on which
the point intersects (or almost intersects) is paired to the blob, which
determines the blobs visibility at a given body orientation. A warning is
provided if a blob does not (almost) intersect a surface.

A projection of the body at any position and orientation is created with the
`update(X, Q)` method. The arguments are absolute position and
orientation from the original position, not from the last update position.

Blob and surface mesh data can be accessed via `blobs` and `vectors`
attributes.

An interactive 3D projection of the geometry at the current projected position
and orientation can be viewed by calling the `plot()` method. 

## Projection

The 3-dimensional body can be projected onto any 2-dimensional viewpoint. 
Each `View()` class represents a different viewpoint, defined by 3 Euler angle
rotations  $(\psi, \theta, \phi)$ pointing towards the local coordinate origin. 
The initial orientation(at Euler angles 0,0,0) aligns with the X and Y axis 
(looking downwards from east). Other common view angles include:
- Top view facing forward: (-$\pi$/2, 0, 0)
- East view: (0, 0, $\pi$/2)
- West view: ($\pi$, 0, $\pi$/2)
- Front view: ($\pi$/2, 0, $\pi$/2)

Only surfaces and blobs with a normal vector pointing towards the camera view
are shown in the 2-dimensional projection. In order to avoid data at very acute
surface angles, a threshold of the dot product of the surface normal vector, and
viewing angle vector is specific, with a default value of 0.01. 

Projected 2-dimensional blob locations are accessed by the `get_blobs()` method,
and 2D surface mesh data is accessed via the `get_mesh()` method. 

### Blob data

A 2D view of blob data is stored in the `BlobsFrame()` class, with the attributes
- `n`: number of blobs in frame
- `points`: (2x$n$) array of 2-dimensional coordinates
- `diameters`: ndarray of blob diameters.

This class can be extended to include additional blob data extracted from image
processing (such as ovality, colour). 

The BlobFrame provides a method to pass *observed* and *projected* blob data
during iterative spatial matching. 

## Spatial match

When using 2D blob data to track bodies (such as high-speed 2D camera images),
the state observables are three position coordinates and four rotational
quaternions, so that $\mathbf{y}_\mathrm{image} = [x, y, z, q0, q1, q2, q3]^\mathrm{T}$. 

More than one different view angle is required to determine the full set of
observations (such as top and east view).

For each image, the *measured* blob locations are compared with the *projected* 
blob locations. The observation vector $\mathbf{y}$ is then iteratively varied until
the best match between measured and projected blobs is achieved. The
algorithm and cost function for this matching algorithm could be improved, and
is under development. 

Also under development is a dynamically prescribed measurement uncertainty
vector $[R]_\mathrm{image}$ which is a function of the correlation between the
final match between measured and projected blob data. 

## Plotting

Three main types of plots are available:
- `PlotMatch`: Shows the match between measured and projected 2D blob locations.
- `PlotTrack`: Tracks the state vector for each update step
- `BodySTL.plot()`: Plots the current body position and orientation in an
    interactive 3D plot.






