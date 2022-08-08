""" Observation and measurment functions
"""

from typing import List, Tuple
import numpy as np

from motiontrack.custom_types import ObservationGroup

def get_all_measurements(obs: List[ObservationGroup], x_pr_dict: dict)\
        -> Tuple(np.array, np.array):
    """
    Get measurements from multiple groups

    Gets measurements values and uncertanties from one or more observation
    groups, and combines into vectors.

    Parameters
    ----------
    obs : List[ObservationGroup]
        List of observation groups
    x_pr_dict : dict
        A priori state estimate used to assist with measurements (such as image tracking)

    Returns
    -------
    y : np.array
        Array of measurement values
    tau : np.array
        Array of observation uncertanties
    """
    y_all = np.empty(0)
    tau_all = np.empty(0)
    for ob in obs:
        _, y, tau = ob.next_measurement(x_pr_dict)
        y_all = np.concatenate([y_all,y])
        tau_all = np.concatenate([tau_all,tau])
    return y_all, tau_all

def get_all_observables(obs: List[ObservationGroup], x_dict: dict)\
        -> np.array:
    """
    Calculate observables

    Calculates observable values from state vector for all observable groups

    Parameters
    ----------
    obs : List[ObservationGroup]
        List of ObservationGroup instances to use for observation
    x_dict : dict
        Dictionary of state vector in form {<name>:value}

    Returns
    -------
    z : np.array
        Array of observation values
    """
    z_all = np.empty(0)
    for ob in obs:
        z = ob.calc_observable(x_dict)
        z_all = np.concatenate([z_all,z])
    return z_all

def get_next_obs_group(obs_all: List[ObservationGroup], t: float, dt_min: float)\
        -> Tuple[List[ObservationGroup],float]:
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
    t_next : Time of the next measurement
    """
    ts = []
    for ob in obs_all:
        t.append(ob.get_next_t())
    t_min = np.min(t)
    obs_next = [obs_all[i] for i in range(len(obs_all)) if ts[i]-t_min<dt_min]
    return obs_next


