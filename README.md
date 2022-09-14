# USQ motion analysis codes

A selection of Python tools used to analyse object motion, intended predominantly for experimental motion data.

![alt_text](/docs/img/tracking.png?raw=true "tracker_outline")
![alt_text](/docs/img/mesh.png?raw=true "tracker_outline")
![alt text](/docs/img/top.png?raw=true "tracker_outline")
![alt text](/docs/img/east.png?raw=true "tracker_outline")

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

## Tests

The tests directory contains scripts to test the functionality of individual modules, and should be used when modifying core modules. 

## Examples

Examples of how the different packages integrate to track dynamic systems.
Includes

- 3DoF tracking from XYZ data
- 6DoF tracking from XYZ and quaternion data
- 6DoF tracking from 2-D projected blob data



