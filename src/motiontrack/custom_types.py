""" Custon types for MotionTrack package.
"""

from typing import List
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
        self.ob_names = []
        self.index = -1
        self.z_history = []
        self.y_history = []
        self.t_history = []
        self.tau_history = []

        if ob_names is None:
            ob_names = [self.name+"_"+str(i) for i in range(self.size)]

    # <Override>
    def _next_measurement(self, x_pr_dict: dict)\
            -> tuple(float, np.array, np.array):
        """ Placeholder for update function.
        Custom function takes a prior state estimate and returns:
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
        """
        return 0

    # <Override>
    def _calc_observables(self, x_dict: dict) -> np.array:
        """ Placeholder function to get observables from state vector.
        State vector variables are refereced by dictionary key.
        """
        return np.array(0)


    def next_mesurement(self, x_pr_dict: dict):
        """ Public method to get next measurement"""
        t, y, tau = self._next_measurement(x_pr_dict)
        self.index += 1
        self.t_history.append(t)
        self.y_history.append(y)
        self.tau_history.append(tau)
        return t, y, tau

    def get_next_t(self) -> float:
        """ Public method to get time of next observation """
        t_next = self._get_next_t()
        return t_next

    def calc_observable(self, x_dict: dict) -> np.array:
        """
        Calculate observable

        Parameters
        ----------
        x_dict : dict
            Dictionary of state values in format {'name':value}

        Returns
        -------
        z : np.array
            Vector of observation values
        """

        z = self._calc_observables(x_dict)
        self.z_history.append(z)
        return z

    def get_tau(self):
        """ Return measurement uncertainty """
        return self.tau_history[-1]

    def get_y(self):
        """ Return measurement vector """
        return self.y_history[-1]

    def get_t(self):
        """ Return current time """
        return self.t_history[-1]
