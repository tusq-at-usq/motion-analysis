#!/usr/bin/python3.8

"""Unscented Kalman Filter for motion tracking
"""

from typing import Callable, Tuple, Optional
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
#  from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from dynamicsystem.system_class import DynamicSystem

class UKF:
    def __init__(self,
                 dsys: DynamicSystem,
                 dt_int_max: float):
        self.dsys = dsys
        self.dt_int_max = dt_int_max
        self.kf = None

    def _fx(self,X,dt):
        # Note: Inputs not yet supported
        X_, t_ = self.dsys.integrate(dt, X, dt_max=self.dt_int_max)
        return X_[-1,:]

    def initialise(self,
                   x_0: np.array,
                   P: Optional[np.array]=None,
                   Q: Optional[np.array]=None,
                   sigmas: Optional[np.array]=None):
        """
        Initialise the Unscented Kalman Filter instance

        Parameters
        ----------
        x_0 : np.array
            Initial state vector
        P : np.array (optional)
            Initial covariance vector
        Q : np.array (optional)
            Process uncertainty vector
        sigmas : np.array
            Override for the default UKF sigma locations and weights
        """
        if sigmas is None:
            sigmas = MerweScaledSigmaPoints(self.dsys.get_nx(),
                                            alpha=.1,
                                            beta=2.,
                                            kappa=-1)
        kf = UnscentedKalmanFilter(dim_x=self.dsys.get_nx(),
                                   dim_z=0,
                                   dt=0,
                                   hx=None,
                                   fx=self._fx,
                                   points=sigmas)
                                   #  x_mean_fn=self.x_mean_fn)
        if P is None:
            P = np.eye(self.dsys.get_nx())*1

        if Q is None:
            Q = np.eye(self.dsys.get_nx())*0.01

        kf.x = x_0 # Initial state
        kf.P = P
        kf.Q = Q
        kf.x = x_0
        self.kf = kf

    def predict(self,
                dt: float,
                nz: int, _hx:
                Callable,
                Q: Optional[np.array]) -> Tuple[np.array, np.array]:
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
        Q : np.array
            Process uncertainty matrix

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
        return self.kf.x, self.kf.P

