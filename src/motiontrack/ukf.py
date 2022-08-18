#!/usr/bin/python3.8

"""Unscented Kalman Filter for motion tracking
"""

from typing import Callable, Tuple, Optional
import numpy as np
from scipy.linalg import expm
import quadpy
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from dynamicsystem.system_class import DynamicSystem

class UKF:
    def __init__(self,
                 dsys: DynamicSystem,
                 dt_int_max: float,
                 normalise_quaternions: bool = False):
        self.dsys = dsys
        self.dt_int_max = dt_int_max
        self.kf = None
        self.normalise_quaternions = normalise_quaternions
        if normalise_quaternions:
            self._normalise_q = self.create_q_norm_fn()
        else:
            self._normalise_q = None

    def _fx(self,X,dt):
        # Note: Inputs not yet supported
        X_, t_ = self.dsys.integrate(dt, X, dt_max=self.dt_int_max)
        return X_[-1,:]

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

    def initialise(self,
                   x_0: np.array,
                   P: Optional[np.array],
                   Q: Optional[np.array]=None,
                   sigmas: Optional[np.array]=None):
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
        sigmas : np.array
            Override for the default UKF sigma locations and weights
        """
        if sigmas is None:
            sigmas = MerweScaledSigmaPoints(self.dsys.get_nx(),
                                            alpha=.1,
                                            beta=2.,
                                            kappa=-1)

        if self.normalise_quaternions:
            def _x_mean_fn(sigmas, Wn):
                x = np.zeros(len(sigmas[0]))
                for i,_ in enumerate(sigmas):
                    for j,_ in enumerate(sigmas[i]):
                        x[j] += sigmas[i,j]*Wn[i]
                x = self._normalise_q(x)
                return x
        else:
            _x_mean_fn=None

        kf = UnscentedKalmanFilter(dim_x=self.dsys.get_nx(),
                                   dim_z=0,
                                   dt=0,
                                   hx=None,
                                   fx=self._fx,
                                   points=sigmas,
                                   x_mean_fn = _x_mean_fn
                                   )
        kf.x = x_0
        kf.P = P
        kf.Q = Q
        kf.x = x_0
        self.kf = kf

    def predict(self,
                dt: float,
                nz: int,
                _hx: Callable,
                Q: Optional[np.array] = None,
                _z_mean_fn: Optional[Callable] = None) -> Tuple[np.array, np.array]:
        """
        Filter predict step: Takes a new timestep and set of observables, and
        updates the filter prediction

        Parameters
        ----------
        dt : float
            Timestep for prediction and update
        nz : int
            Number of observables
        _hx : Callable
            Obsevable function
        Q : np.array (optional)
            Process uncertainty matrix
        _z_mean_fn : Callable (optional)
            Custom function to evaluate mean of sigma points passed through hx

        Returns
        ----------
        x_pr : np.array
            A priori state vector
        z_pr : np.array
            A priori obseravble vector
        """
        self.kf._dt = dt
        self.kf.nz = nz
        self.kf.hx = _hx
        if Q is not None:
            self.kf.Q = Q

        if _z_mean_fn is not None:
            self.kf.z_mean_fn = _z_mean_fn

        self.kf.predict()
        return self.kf.x, _hx(self.kf.x)

    def update(self, y: np.array, R: np.array) -> Tuple[np.array, np.array]:
        """
        Main filter update step

        Takes a measurement and time-step, and updates the
        filter state and covariance.

        Parameters
        ----------
        y : np.array
            Measurement vector
        R : np.array
            Uncertainty matrix

        Returns
        ----------
        x : np.array
            State estimate
        P : np.array
            Covariance estimate
        """
        self.kf.R = R
        self.kf.update(y)
        if self.normalise_quaternions:
            self.kf.x = self._normalise_q(self.kf.x)
        return self.kf.x, self.kf.P

def custom_process_noise(A: np.array,
                         Q_c: np.array,
                         dt: float,
                         density: float) -> np.array:
    """
    Create a disrete-time process noise matrix based off a linear system
    matrix and continuous noise matrix.

    The discret noise matrix is calculated as
    Q_d = int_0^dt F @ Q_c @ F.T dt

    Parameters
    ----------
    A : np.array
        Linear system tranition matrix (can be an estimate)
    Q_c : np.array
        Continuous-time process noise matrix (normally sparse)
    dt_ : float
        Discrete time interval
    density : float
        density (magnitude) of the noise

    Returns
    -------
    Q_d : np.array
        Discrete-time process noise matrix
    """

    def _eval(ts):
        res = []
        for t in ts:
            res.append(expm(A*t)@Q_c@expm(A*t).T)
        return np.moveaxis(res, 0, 2)

    Q_d, err = quadpy.quad(_eval, 0, dt)
    return Q_d








