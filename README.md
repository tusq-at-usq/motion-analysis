# USQ motion analysis codes

A selection of Python tools used to analyse object motion, predominantly to extract aerodynamic acceleration data. 

The broad idea is to combine many measurements into a Bayesian filter to estimate the most probably state at each point in time.

## Contents

This package is separated into 4 parts:
- [`dynamicsystem`](/src/dynamicsystem/README.md) provides a dynamic system builder and container, with associated tools.
- [`motiontrack`](/src/motiontrack/README.md) handles body geometry, body camera projection, and overhead functions to combine multiple observations.
- [`kalmanfilter`](/src/kalmanfilter/README.md) provides nonlinear Kalman filters and smoothers.
- [`cameracalibration`](/src/camera_cal/README.md) provides individual and stereo camera calibration utilities 

In typical usage, these four parts would interact as below.
![alt text](/docs/img/block_diagram.png)

## Install

Clone the repository:
`git clone https://github.com/tusq-at-usq/motion-analysis`

Install the package (recommended in virtual environment):
`python3 -m pip install ./`
or if you plan to make changes, or wish to view the example codes, it is recommended to install using pip in editable mode:
`python3 -m pip install -e ./`

Please [contact Andy](mailto:andrew.lock@usq.edu.au) if you encounter any install issues.

## Dependencies

All dependencies should be installed as part of the pip installation.

*___Note___: This package requires the graphics package `PyQt6`, and therefore also requires Python version >= 3.10.
It is possible to use older Python versions with `PyQt5`, however often this conflicts with other packages also using `PyQT5`. 
If you wish to install using an older version of Python, please [contact Andy](mailto:andrew.lock@usq.edu.au) for details how.*

## Documentation 

Documentation for the theory and usage of this package is provided in the  [`/docs`](/docs/).

In addition, a short description of each package is provided in the `README.md` of each package. 

## Tests

The [`tests/`](/tests) directory provides a suite of test codes designed to run with [pytest](https://docs.pytest.org/).

Note the test modules are currently under active development.

## Examples

Examples are provided in [`examples/`](/examples/). Currently the examples include:
- Stereo tracking of a cube dropped in free fall ([drop test](/examples/drop_example))

More to come...

## Roadmap

motion-analysis is being developed along the following roadmap.

* [ ] Create block-based dynamic system builder
* [ ] Provide more examples
* [ ] Provide test modules and auto-test on push




