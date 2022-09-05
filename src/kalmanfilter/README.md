
# Kalman filter

## Types

This package provides an Extended Kalman filter (EFK) and Unscented Kalman
filter for motion tracking of nonlinear systems.
For tracking position and orientation using quaternions, an EFK is recommended so that the quaternion linear state transition matrix can
be exploited. For other applications, a UKF may be preferable. 

An EFK uses a local linearisation of the state transition matrix $[F]$ and observation function $[H]$ using the Jacobian. A UKF accounts for nonlinearity in the 
system model by passing sample points through the dynamic functions
$f(\boldsymbol{x})$, and observable functions $h(\boldsymbol{x})$, and then reconstruct the Gaussian transformation. 

## Observations

This package extends a typical Kalman filter loop to allow variations to the number and type of observables. It also allows dynamic variations to the process noise vector Q, using a local linearisation of the dynamic functions from the system Jacobian.

Multiple different types of observables can be incorporated with different (and
varying) time intervals. An interface class `ObservationGroup` provides a
standardised interface to the UKF, and can be inherited by any observation type
(e.g. image data from multiple agles, IMU data, sensor data).

At each time step, the set of observables are queried to find the next subset of
groups with measurements at the next timestep. The set of observable functions
$\boldsymbol{h}(\boldsymbol{x}) _k$ specific to the next timestep are used for
the filter. Each `ObservableGroup` instance specifies the measurement
uncertainty $\tau$ for each measurement (which can be dynamically defined using
the assessed quality of the measurement, such as in the case of image matching),
and this is combined to a system measurement noise $R$ matrix. 

## Process noise

A function to compute the process noise matrix specific to the system is
provided.
The process noise matrix $[\boldsymbol{Q}]$ can be reconstructed at each timestep. The process noise assumes a continuous white noise spectrum, and is calculated by
$$ Q = \int_0^{\delta t} F(t) Q F(t)^{-1} dt $$
where $F(t)=e^{A t}$ is the discrete state transition matrix, and $A$ is the continuous-time system matrix, calculated from the system Jacobian $J(\boldsymbol{x})$. 

The `DynamicSystem` class
incorporates a method to evaluate the Jacobian for a given state vector.
The integral is calculated efficiently for any $A$ using Gaussian quadrature. 

The filters are based on code from the package `filterpy`:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

A useful resource for understanding Kalman filters and FilterPy is [here](https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)
