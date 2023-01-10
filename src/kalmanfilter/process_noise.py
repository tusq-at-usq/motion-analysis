""" Tools associated with Kalman filters
"""

import numpy as np
from scipy.linalg import expm
from scipy import integrate

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

    def eval(t):
        return expm(A*t)@(Q_c*density)@expm(A*t).T

    Q_d = integrate.quad_vec(eval, 0.0, dt)[0]
    return Q_d





