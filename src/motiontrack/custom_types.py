""" Custon types for MotionTrack package.
"""

from typing import List, Tuple, Callable
import numpy as np

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
        self.index = -1
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


    # <Override>
    def _get_next_t(self) -> float:
        """ Placeholder function to get the time of next measurement.
        Used for scheduling.
        Returns float of next observation time.
        A value of -1 implies no further measurements.
        """
        return 0

    # <Override>
    def _create_ob_fn(self, x_dict: dict) -> Callable:
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

    def create_ob_fn(self, x_dict: dict) -> Callable:
        """
        Create an observable function used in the Kalman filter,
        of the form z = h(x)

        Parameters
        ----------
        x_list : dict
            Dictionary of state values in format {'name':value}

        Returns
        -------
        hx : function
            Observable function z = h(x)
        """

        hx = self._create_ob_fn(x_dict)
        return hx

    def residual(self,y1,y0):
        """ Placeholder for residual calculation.
            Overide if using quaternions
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
