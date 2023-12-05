""" Observation 
"""

from typing import List, Tuple, Callable
import numpy as np

class Observation:
    """ Template for an observation group, to be inherited by child observation classes.

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
        self.z_history = []
        self.y_history = []
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
        y, tau = self._next_measurement(x_pr, x_dict)
        self.y_history.append(y)
        self.tau_history.append(tau)
        return y, tau

    def get_next_t(self) -> float:
        """ Public method to get time of next observation.
        A value of -1 implies there are no further measuremnents"""
        t_next = self._get_next_t()
        return t_next

    @property
    def next_t(self):
        return self.get_next_t()

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

    def get_nz(self):
        """ Return number of observables """
        return self.size


def get_all_measurements(obs: List[Observation],
                         x_pr: np.array,
                         x_dict: dict) -> Tuple[np.array, np.array]:
    """
    Get measurements from multiple groups

    Gets measurements values and uncertanties from one or more observation
    groups, and combines into vectors.

    Parameters
    ----------
    obs : List[Observation]
        List of observation groups
    x_pr : np.array
        State vector`
    x_dict : dict
        Dictionary of state vector in format {<name> : <index>}

    Returns
    -------
    y : np.array
        Array of measurement values
    R : np.array
        Matrix of observation uncertanties
    """
    y_all = np.empty(0)
    tau_all = np.empty(0)
    for ob in obs:
        y, tau = ob.next_measurement(x_pr, x_dict)
        y_all = np.concatenate([y_all,y])
        tau_all = np.concatenate([tau_all,tau])
        R = np.eye(len(tau_all))*tau_all
    return y_all, R

def create_observable_function(obs: List[Observation],
                               x_dict: dict,
                               u_dict: dict = {}) -> np.array:
    """
    Create combined observable function

    Creates a combined observable function from all observable groups
    for the next update time.

    Parameters
    ----------
    obs : List[Observation]
        List of Observation instances to use for observation
    x_dict : dict
        Dictionary of states in state vector, in form {<name>:index}
    u_dict : dict
        Dictionary of inputs in state input, in form {<name>:index}

    Returns
    -------
    hx : Callable
        Combined observable function
    """
    hx_groups = [ob.create_ob_fn(x_dict, u_dict) for ob in obs]

    def hx(x: np.array, u: np.array):
        z = [hx_i(x, u) for hx_i in hx_groups]
        z = np.array(np.concatenate(z).flat)
        return z
    return hx

def create_residual_function(obs: List[Observation]):
    """ Combine the observation residual functions of each group
    """

    def residual_fn(y1, y0):
        nz_count = 0
        residual = []
        for ob in obs:
            nz = ob.get_nz()
            residual.append(ob.residual(y1[nz_count:nz_count+nz],
                                        y0[nz_count:nz_count+nz]))
        nz_count += nz
        return np.array(residual)
    return residual_fn

def get_next_obs_group(obs_all: List[Observation], t: float, dt_min: float)\
        -> Tuple[float, List[Observation], int]:
    """
    Get sub-group of observations at next observation time

    Observations may not be time-synced, and may have different measurement
    time intervals. This function queries all observation groups, and returns
    a list of the groups relevant to the next observation.

    Parameters
    ----------
    obs_all : List[Observation]
        List of all observation groups
    t : float
        Previous measurement time
    dt : float
        Tolerance to accept multiple measurements at the same time

    Returns
    -------
    obs_next : List[Observation]
        List of observation groups relevant to the next measurement time
    t_next : float 
        Time of the next measurement
    nz : int
        Number of observables
    """

    ts = []
    for ob in obs_all:
        ts.append(ob.get_next_t())
    obs_next = [obs_all[i] for i in range(len(obs_all)) if np.abs(ts[i]-t)<dt_min]
    nz = np.sum([ob.get_nz() for ob in obs_next])
    return obs_next, nz


def get_next_t(data_iterators):
    ts = []
    for d in data_iterators:
        ts.append(d.next_t)
    t_min = np.min(ts)
    return t_min

