# `motiontrack`

## `example_vehicle_projection.py`

An example of object projection with different view angles.

## `example spatial match.py`

A fully working example that demonstrates the position and oritentation solver.

Models a 1x1m cube body with 4 blobs per surface (randomly allocated).
Has two views: from above and west.

Simulates imperfect  data with:
* random Gaussian error for each data point
* 2 blobs missing for each view

Initial position and orientation has random errors:
* position error between [0,10cm]
* rotation errors betweem [0,pi/8]

