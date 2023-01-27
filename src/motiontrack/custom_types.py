""" Custon types for MotionTrack package.
"""

from typing import List, Tuple, Callable
import numpy as np
import cv2 as cv

class ObservationGroup:
    """ Template for an observation group, to be inherited by custom functions.

    Naming convention:
        z:      Calculated observation
        y:      Measurment array
        tau:    Uncertainty array associated with measurement
        x_pr:   A priori state estimate (guess) to aid observation

    Parent classes should define the placeholder methods:
        1. _next_measurement
        2. _get_next_t
        3. _calc_observable
    See placeholder methods for details.

    """

    def __init__(self,
                 name: str, # Name of observable group
                 size: int, # Number of individual observations
                 ob_names: List[str]=None # List of observation names
                 ):

        self.name = name
        self.size = size
        self.ob_names = ob_names
        self.index = 0
        self.z_history = []
        self.y_history = []
        self.t_history = []
        self.tau_history = []

        if ob_names is None:
            ob_names = [self.name+"_"+str(i) for i in range(self.size)]

    # <Override>
    def _next_measurement(self, x_pr: np.array, x_dict: dict)\
            -> Tuple[float, np.array, np.array]:
        """ Placeholder for update function.
        Custom function takes a priori state estimate and returns:
        float t: new time
        np.array y: new observable vector
        np.array tau: uncertainty vector associated with new measurments
        """
        t=0
        z=np.array(0)
        tau = np.array(0)
        return t, z, tau

    def change_size(self, nz):
        """
        Change the number of observations 
        (for dynamically defined observations)

        Parameters
        ----------
        nz : int
            Updated number of observations
        """
        self.size = nz

    # <Override>
    def _get_next_t(self) -> float:
        """ Placeholder function to get the time of next measurement.
        Used for scheduling.
        Returns float of next observation time.
        A value of -1 implies no further measurements.
        """
        return 0

    # <Override>
    def _create_ob_fn(self, x_dict: dict, u_dict: dict, x_pr: np.array) -> Callable:
        """ Placeholder function to get observables from state vector.
        State vector variables are refereced by dictionary key.
        """
        hx = lambda x: np.array(0)
        return hx

    def next_measurement(self, x_pr: np.array, x_dict: dict):
        """
        Get next measurement values from observation group

        Parameters
        ----------
        x_pr : np.array
            A priori state vector
        x_dict : dict
            State index dictionary with format {<name>: <index>}
        """
        t, y, tau = self._next_measurement(x_pr, x_dict)
        self.index += 1
        self.t_history.append(t)
        self.y_history.append(y)
        self.tau_history.append(tau)
        return y, tau

    def get_next_t(self) -> float:
        """ Public method to get time of next observation.
        A value of -1 implies there are no further measuremnents"""
        t_next = self._get_next_t()
        return t_next

    def create_ob_fn(self, x_dict: dict, u_dict: dict, x_pr: np.array = np.empty(0)) -> Callable:
        """
        Create an observable function used in the Kalman filter,
        of the form z = h(x)

        Parameters
        ----------
        x_dict : dict
            Dictionary of state values in format {'name':value}
        u_dict : dict
            Dictionary of input values in format {'name':value}

        Returns
        -------
        hx : function
            Observable function z = h(x)
        """

        hx = self._create_ob_fn(x_dict, u_dict, x_pr)
        return hx

    def residual(self,y1,y0):
        """ Placeholder for residual calculation.
            May want to override for quaternions
            """
        return y1-y0

    def get_tau(self):
        """ Return measurement uncertainty """
        return self.tau_history[-1]

    def get_y(self):
        """ Return measurement vector """
        return self.y_history[-1]

    def get_t(self):
        """ Return current time """
        if len(self.t_history) == 0:
            return -1
        return self.t_history[-1]

    def get_nz(self):
        """ Return number of observables """
        return self.size

class CameraCalibration:
    def __init__(self, mtx, dist, R=np.eye(3), T=np.array([0.,0.,0.]), parallel=0, scale=1):
        self.mtx = mtx
        self.dist = dist
        self.R = R # Rotation from camera 2 to camera 1
        self.T = T # Translation from camera 2 to camera 1
        self.R_L = np.eye(3) # Rotation from C1 openCV to local coordiantes
        self.parallel = parallel
        self.scale = scale


    def init(self):
        self.dist = np.pad(self.dist[0],(0,14-self.dist.shape[1]))
        tau_x = self.dist[12]
        tau_y = self.dist[13]
        R_ = np.array([[np.cos(tau_y), np.sin(tau_y)*np.sin(tau_x), -np.sin(tau_y)*np.cos(tau_x)],
                      [0, np.cos(tau_x), np.sin(tau_x)],
                      [np.sin(tau_y), -np.cos(tau_y)*np.sin(tau_x), np.cos(tau_y)*np.cos(tau_x)]])
        self.R_cor = np.array([[R_[2,2], 0, -R_[0,2]],
                               [0, R_[2,2], -R_[1,2]],
                               [0, 0, 1]]) @ R_

    def set_intrinsic(self, mtx, dist, scale=1):
        self.mtx = mtx
        self.dist = dist
        self.scale = scale
        self.init()

    def set_extrinsic(self, R, T):
        self.R = R
        self.T = T
        
    def set_all(self, mtx, dist, R, T, scale=1):
        self.set_intrinsic(mtx, dist, scale)
        self.set_extrinsic(R, T)


    # Outdated opencv implementation
    #  def _project(self, X, R_C=None):
        #  if R_C is None:
            #  R_C = self.R
        #  Y = cv.projectPoints(self.R_L@X, R_C, self.T, self.mtx, self.dist)
        #  Y = Y[0].reshape(-1,2)
        #  return Y


    def project_wo_distortion(self, X):
        X_ = self.R_L@X
        P_w = np.vstack((X_,np.full(X_.shape[1],1)))
        RT = np.hstack((self.R,self.T.reshape(-1,1)))
        P_c = RT@P_w
        Z_c = P_c[2]
        xy = P_c/Z_c
        Y = self.mtx@xy
        Y /= Y[2]
        Y = Y[0:2].T.reshape(-1,2)
        return Y

    def _project(self, X, R_C=None, T = None):
        if R_C is None:
            R_C = self.R
        if T is None:
            T = self.T
        X_ = self.R_L@X
        P_w = np.vstack((X_,np.full(X_.shape[1],1)))
        RT = np.hstack((R_C,T.reshape(-1,1)))
        P_c = RT@P_w
        Z_c = P_c[2]
        if self.parallel:
            xy = P_c/self.scale
            xy[2] = np.full(xy.shape[1],1)
        else:
            xy = P_c/Z_c
        r2 = (xy[0]**2 + xy[1]**2)

        dist = self.dist
        t1 = (1 + dist[0]*r2 + dist[1]*r2**2 + dist[4]*r2**3)/ \
            (1 + dist[5]*r2 + dist[6]*r2**2 + dist[7]*r2**3)
        tx1 = 2*dist[2]*xy[0]*xy[1]
        ty2 = 2*dist[3]*xy[0]*xy[1]
        tx2 = dist[3]*(r2 + 2*xy[0]**2)
        ty1 = dist[2]*(r2 + 2*xy[1]**2)

        xy_cor = np.full(xy.shape,1.)
        xy_cor[0,:] = xy[0]*t1 + tx1 + tx2 + dist[8]*r2 + dist[9]*r2**2
        xy_cor[1,:] = xy[1]*t1 + ty1 + ty2 + dist[10]*r2 + dist[11]*r2**2

        xy_cor2 = self.R_cor@xy_cor
        Y = self.mtx@xy_cor2
        Y /= Y[2]
        Y = Y[0:2].T.reshape(-1,2)
        return Y

    def project(self, X, R_C=None, T = None):
        return self._project(X, R_C, T)
