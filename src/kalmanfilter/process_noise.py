""" Tools associated with Kalman filters
"""

import numpy as np
from scipy.linalg import expm
import quadpy

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
        Linear dynamical system matrix (can be an estimate)
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
            res.append(expm(A*t)@(Q_c*density)@expm(A*t).T)
        return np.moveaxis(res, 0, 2)

    Q_d, err = quadpy.quad(_eval, 0, dt)
    return Q_d





