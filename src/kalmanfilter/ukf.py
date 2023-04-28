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
from copy import deepcopy
from scipy import linalg
from scipy.linalg import expm
from scipy.linalg import cholesky
from filterpy.stats import logpdf
from filterpy.common import pretty_str
from filterpy.kalman import MerweScaledSigmaPoints

from dynamicsystem.system_class import DynamicSystem

from motiontrack.utils import *

def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    This works in conjunction with the UnscentedKalmanFilter class.


    Parameters
    ----------

    sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.

    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.

    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    residual_fn : callable (x, y), optional

        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                y = y % (2 * np.pi)
                if y > np.pi:
                    y -= 2*np.pi
                return y

    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing throgh the transform.

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    kmax, n = sigmas.shape

    try:
        if mean_fn is None:
            # new mean is just the sum of the sigmas * weight
            x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
        else:
            x = mean_fn(sigmas, Wm)
            x_ = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
    except:
        print(sigmas)
        raise

    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :]
        P = np.dot(y.T, np.dot(np.diag(Wc), y))
    else:
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)

class UnscentedKalmanFilter():

    """ Implements an Unscented Kalman filter (UKF). You are responsible for
    setting the various state variables to reasonable values; the defaults
    will  not give you a functional filter.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    t can also fail silently - you can end up with matrices of a size that
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
        #  self.points = points

        dim_x = dsys.get_nx()
        self.dim_x = dim_x

        self.x = np.zeros((dim_x, 1)) # state
        self.P = np.eye(dim_x)        # uncertainty covariance

        self.z = np.empty(0)
        #  z = np.array([None]*self.dim_z)
        #  self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.msqrt = cholesky

        #  self.sigma_points_fn = MerweScaledSigmaPoints(dsys.get_nx(), alpha=.1, beta=2., kappa=-1)
        self.sigma_points_fn = MerweScaledSigmaPoints(dsys.get_nx(), alpha=0.001, beta=2., kappa=0.)
        self._num_sigmas = self.sigma_points_fn.num_sigmas()
        self.Wm = self.sigma_points_fn.Wm
        self.Wc = self.sigma_points_fn.Wc

        self.residual_x = np.subtract
        self.residual_y = np.subtract

        # DEBUG
        # Testign custom residual function
        def residual_x(x_, x):
            dx = x_ - x
            #  dx[3:7] = dx[3:7]/np.linalg.norm(dx[3:7])
            x[3:7] = x[3:7]/np.linalg.norm(x[3:7])
            x_[3:7] = x_[3:7]/np.linalg.norm(x_[3:7])
            dx[3:7] = quaternion_subtract(x_[3:7], x[3:7])
            #  dx[3:7] = dx[3:7]/np.linalg.norm(dx[3:7]) - np.array([1, 0, 0, 0])
            #  print(dx[3:7])
            return dx
        #  self.residual_x = residual_x

        def mean_fn(sigmas, Wm):
            Wm = np.array(Wm)/np.sum(Wm)
            x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])
            qs = [s[3:7] for s in sigmas]
            q_av = quaternion_weighted_av(qs,Wm)*-1
            x[3:7] = q_av[:]
            return x
        
        #  self.mean_fn = mean_fn
        self.mean_fn = None



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
        dx = 1e-6

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

    def compute_process_sigmas(self, dt, x, P, u = np.array([])):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        fx = self._fx

        # calculate sigma points for given mean and covariance
        # DEBUG:
        sigmas = self.sigma_points_fn.sigma_points(x, P)
        #  sigmas[:,3:7] = sigmas[:,3:7]/np.linalg.norm(sigmas[:,3:7], axis=1)[:,None]
        
        self.sigmas_f = np.zeros((self._num_sigmas, self.dsys.get_nx()))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, u, dt)

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_y(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    def predict(self,
                dt: float,
                x: np.array,
                P: np.array,
                Q: np.array,
                u: np.array = np.array([])):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Note: Not yet configured with inputs

        Parameters
        ----------
        dt : float
            Time step of the tracking iteration
        x : np.array
            state vector
        P : np.array
            state covariance matrix
        Q : np.array
            Process noise matrix
        u : np.array (optional)
            input vector
        """

        self.compute_process_sigmas(dt, x, P, u)

        self.x, self.P = unscented_transform(self.sigmas_f, self.Wm, self.Wc, Q,
                                             residual_fn=self.residual_x, mean_fn = self.mean_fn)
        if self.quaternions:
            self.x = self._normalise_q(self.x)

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        J = self.dsys.J_np(self.x_prior, u)
        return self.x_prior, self.P_prior

    def update(self,
               y: np.array,
               R: np.array,
               _hx: Callable,
               u: np.array = np.array([])):
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
        u : np.array, optional
            Input vector
        """

        if y is None or len(y) == 0:
            self.z = np.array([[None]]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(_hx(s, u))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        zp, self.S = unscented_transform(self.sigmas_h, self.Wm, self.Wc, R)
        self.SI = np.linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = np.dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_y(y, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))
        if self.quaternions:
            self.x = self._normalise_q(self.x)

        # save measurement and posterior state
        self.z = deepcopy(y)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        return self.x, self.P

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
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
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def rts_smoother(self, Xs, Ps, Qs, dts, quaternions=True):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """
        #pylint: disable=too-many-locals, too-many-arguments

        n = len(Xs)
        if not all(n_i == n for n_i in [len(Ps),
                                        len(Qs),
                                        len(Xs),
                                        len(dts)]):
            print("ERROR: RTS smoother inputs are not same length")

        n = len(Xs)
        dim_x = self.dsys.get_nx()

        UT = unscented_transform

        # smoother gain
        Ks = np.zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = deepcopy(Xs), deepcopy(Ps)
        sigmas_f = np.zeros((num_sigmas, dim_x))

        for k in reversed(range(n-1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.sigma_points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self._fx(sigmas[i], np.array([]), dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, Qs[k])

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * np.outer(z, y)

            # compute gain
            K = np.dot(Pxb, np.linalg.inv(Pb))

            # update the smoothed estimates
            xs[k] += np.dot(K, self.residual_x(xs[k+1], xb))
            if quaternions:
                xs[k] = self._normalise_q(xs[k])
            ps[k] += np.dot(K, ps[k+1] - Pb).dot(K.T)
            Ks[k] = K

        return xs, ps
