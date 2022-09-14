""" Observation and measurment functions
"""

from typing import List, Tuple, Callable
import numpy as np

from motiontrack.custom_types import ObservationGroup

def get_all_measurements(obs: List[ObservationGroup], x_pr: np.array, x_dict: dict)\
        -> Tuple[np.array, np.array]:
    """
    Get measurements from multiple groups

    Gets measurements values and uncertanties from one or more observation
    groups, and combines into vectors.

    Parameters
    ----------
    obs : List[ObservationGroup]
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

def create_observable_function(obs: List[ObservationGroup], x_dict: dict)\
        -> np.array:
    """
    Create combined observabel function

    Creates a combined observanle function from all observable groups
    for the next update time.

    Parameters
    ----------
    obs : List[ObservationGroup]
        List of ObservationGroup instances to use for observation
    x_dict : dict
        Dictionary of states in state vector, in form {<name>:index}

    Returns
    -------
    hx : Callable
        Combined observable function
    """
    hx_groups = [ob.create_ob_fn(x_dict) for ob in obs]

    def hx(x: np.array):
        z = [hx_i(x) for hx_i in hx_groups]
        z = np.array(np.concatenate(z).flat)
        return z
    return hx

def create_residual_function(obs: List[ObservationGroup]):
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

def get_next_obs_group(obs_all: List[ObservationGroup], t: float, dt_min: float)\
        -> Tuple[float, List[ObservationGroup], int]:
    """
    Get sub-group of observations at next observation time

    Observations may not be time-synced, and may have different measurement
    time intervals. This function queries all observation groups, and returns
    a list of the groups relevant to the next observation.

    Parameters
    ----------
    obs_all : List[ObservationGroup]
        List of all observation groups
    t : float
        Previous measurement time
    dt : float
        Tolerance to accept multiple measurements at the same time

    Returns
    -------
    obs_next : List[ObservationGroup]
        List of observation groups relevant to the next measurement time
    t_next : float 
        Time of the next measurement
    nz : int
        Number of observables
    """
    ts = []
    for ob in obs_all:
        ts.append(ob.get_next_t())
    t_min = np.min(ts)
    if t_min == np.Inf:
        return np.Inf, [], 0
    obs_next = [obs_all[i] for i in range(len(obs_all)) if ts[i]-t_min<dt_min]
    nz = np.sum([ob.get_nz() for ob in obs_next])
    return t_min, obs_next, nz


