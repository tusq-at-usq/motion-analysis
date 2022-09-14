# USQ motion analysis codes

A selection of Python tools used to analyse object motion, intended predominantly for experimental motion data. Multiple types of observations (image blobs detection, IMU etc.) can be combined to estimate the state of the system at the current time point. 

Future development will also incorporate data smoothing, so that forces and
moments can be extracted from the motion data. 

![alt_text](/docs/img/example_gif.gif.png?raw=true "mesh_view")
<!-- ![alt_text](/docs/img/mesh.png?raw=true "mesh_view") -->
<!-- ![alt text](/docs/img/top.png?raw=true "top_blob_match") -->
<!-- ![alt text](/docs/img/east.png?raw=true "east_blob_match") -->

## Install

Clone the repository:
`git clone https://github.com/tusq-at-usq/motion-analysis`

Install using pip in editable mode:
`python3 -m pip install -e ./`

Please contact Andy (andrew.lock@usq.edu.au) if you encounter any install issues.

## Contents

This repository contains the following Python packages

* [`motiontrack`](/src/motiontrack/README.md)
* [`blobdetect`](/src/blobdetect/README.md)
* [`dynamicsystem`](/src/dynamicsystem/README.md)
* [`kalmanfilter`](/src/kalmanfilter/README.md)
* (more to be included)

## Dependencies 

Most dependencies should be automatically installed during the package install.
If you encounter unmet dependencies, please contact Andy. 

## Documentation 

Documentation to be added to [`docs`](/docs/).

A description of each package is provided in the `README` of each package. 

## Tests

The tests directory contains scripts to test the functionality of individual modules, and should be used when modifying core modules. 

## Examples

Examples of how the different packages integrate to track dynamic systems.
Include

- 3DoF tracking from XYZ data
- 6DoF tracking from XYZ and quaternion data
- 6DoF tracking from 2-D projected blob data from a cube

## Example structure of tracker design

The below flowchart shows an example structure 

![alt text](/docs/img/motiontrack_diagram.png?raw=true "motiontrack_structure")


