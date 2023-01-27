# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments


"""
Based heavily off the filterpy Kalman filter implementation at:
http://github.com/rlabbe/filterpy

Modified for the particular uses for motion tracking, and to
integrate with other associated packages.

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import sys
from typing import Callable, Optional
import numpy as np
from scipy import linalg
from scipy.linalg import expm
from filterpy.stats import logpdf
from filterpy.common import pretty_str
from dynamicsystem.system_class import DynamicSystem

class ExtendedKalmanFilter():

    """ Implements an extended Kalman filter (EKF). You are responsible for
    setting the various state variables to reasonable values; the defaults
    will  not give you a functional filter.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate vector
    P : numpy.array(dim_x, dim_x)
        Covariance matrix
    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
    F : numpy.array()
        State Transition matrix
    H : numpy.array(dim_x, dim_x)
        Measurement function
    y : numpy.array
        Residual of the update step. Read only.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.
    S :  numpy.array
        Systen uncertaintly projected to measurement space. Read only.
    z : ndarray
        Last measurement used in update(). Read only.
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurment. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Read only.
    Examples
    --------
    See the reference book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self,
                 dsys: DynamicSystem,
                 dt_int_max: float,
                 quaternions: bool = False):

        self.dsys = dsys
        self.dt_int_max = dt_int_max
        self.quaternions = quaternions
        if self.quaternions:
            self._normalise_q = self.create_q_norm_fn()

        dim_x = dsys.get_nx()
        self.dim_x = dim_x

        self.x = np.zeros((dim_x, 1)) # state
        self.P = np.eye(dim_x)        # uncertainty covariance
        self.B = 0                 # control transition matrix
        self.F = np.eye(dim_x)     # state transition matrix
        self.R = np.empty(0)       # state uncertainty
        self.Q = np.eye(dim_x)        # process uncertainty
        self.y = np.empty(0)          # residual

        self.z = np.empty(0)
        #  z = np.array([None]*self.dim_z)
        #  self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step.
        # Because observations are dynamically updated, they are re-sized
        # at each update step.
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        #  self.y = zeros((dim_z, 1))
        self.res = np.empty(0)
        #  self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.S = np.empty(0)
        #  self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty
        self.SI = np.empty(0)

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self._log_likelihood = np.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def _fx(self,X, u, dt):
        # Note: Inputs not yet supported
        X_, t_ = self.dsys.integrate(dt,
                                     X,
                                     dt_max=self.dt_int_max,
                                     u_funs = u,
                                     input_type = 'constant')
        return X_[-1,:]

    def initialise(self,
                   x_0: np.array,
                   P: Optional[np.array],
                   Q: Optional[np.array]=None):
        """
        Initialise the Unscented Kalman Filter instance

        Parameters
        ----------
        x_0 : np.array
            Initial state vector
        P : np.array
            Initial covariance vector
        Q : np.array
            Process uncertainty vector
        """

        self.x = x_0
        self.P = P
        self.Q = Q

    def create_q_norm_fn(self):
        """
        Create function to normalise quaternions in state vector

        Returns
        ----------
        _normalise_q : np.array
            A priori state vector
        """
        q_inds =  np.array([self.dsys.x_dict[q_x] for q_x in ['q0', 'q1', 'q2', 'q3']])
        if len(q_inds) not in [0,4]:
            print("ERROR:",len(q_inds),"quaternions identifiead")
        def _normalise_q(x):
            q_raw = x[q_inds]
            q_norm = q_raw / np.linalg.norm(q_raw)
            for q_i,q_ind in zip(q_norm,q_inds):
                x[q_ind] = q_i
            return x
        return _normalise_q

    def create_H_jacobian(self,
                           x: np.array,
                           u: np.array,
                           n_z: int,
                           hx: Callable,
                           dt: float):
        """
        Internal function to create a Jacobian matrix of the observation
        function.

        Uses the time-step dt to determine the dx of each variable.

        Parameters
        ----------
        x : np.array
            Current state
        u : np.array
            Current state
        dt : float
            Time-step. NOTE: Currently not used
        """
        dx = 1e-8

        x0 = x.copy()
        #  x1 = self._fx(x0,dt)
        z0 = hx(x0, u)

        H = np.empty((n_z, len(x0)))
        for i in range(len(x0)):
            xd_i = x0.copy()
            xd_i[i] += dx
            dzdx = (hx(xd_i, u)-z0)/dx
            H[:,i] = dzdx
        return H

    def predict(self,
                dt: float,
                Q: np.array,
                u = np.array([])):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Note: Not yet configured with inputs

        Parameters
        ----------
        dt : float
            Time step of the tracking iteration
        Q : np.array
            Process noise matrix
        """
        x_priori = self._fx(self.x, u, dt)
        J = self.dsys.J_np(x_priori, u)
        self.F = expm(J*dt)
        self.x = x_priori
        self.Q = Q
        self.P = self.F@self.P@self.F.T + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.F = np.copy(self.F)
        return self.x_prior, self.P_prior, self.F

    def update(self,
               y: np.array,
               R: np.array,
               _hx: Callable,
               H: np.array,
               u: np.array = np.array([]),
               residual: Callable=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.
        Parameters
        ----------
        y : np.array
            measurement for this step.
            If `None`, posterior is not computed
        R : np.array
            measurement uncertainty matrix
        _hx : Callable
            obseravble function
        Hx : np.array
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.
        u : np.array, optional
            Input vector
        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """
        PHT = self.P@(H.T) #np.dot(self.P, H.T)
        self.R = R
        self.S = H@PHT + R
        #  breakpoint()
        self.SI = linalg.inv(self.S)
        self.K = np.dot(PHT, self.SI)

        hx = _hx(self.x, u)
        res = residual(y, hx)
        self.x = self.x + np.dot(self.K, res)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P),I_KH.T) + np.dot(np.dot(self.K, R),self.K.T)
        #  self.P = I_KH@self.P

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.res = res.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        if self.quaternions:
            self.x = self._normalise_q(self.x)
        return self.x, self.P

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """

        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.res, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.res.T, self.SI), self.res)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('res', self.res),
            pretty_str('S', self.S),
            pretty_str('likelihood', self.likelihood),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('mahalanobis', self.mahalanobis)
            ])
