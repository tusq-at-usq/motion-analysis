# pylint: disable=invalid-name, too-many-arguments, too-many-branches,
""" Rauch–Tung–Striebel smoothing

Source:
https://en.wikipedia.org/wiki/Kalman_filter
"""

from typing import List
import numpy as np

def rts_smoother(xs: List[np.array],
                 Ps: List[np.array],
                 x_prs: List[np.array],
                 P_prs: List[np.array],
                 Fs: List[np.array],
                 Qs: List[np.array]):
    """
    Rauch–Tung–Striebel smoothing on batch-processed Kalman filter results

    Note that the inputs for each timestep k are:
        - Kalman filter state vector and covariance: x_(k/k), P_(k/k)
        - The a-priori state and covariance: x_(k/k-1), P_(k/k-1)
        - The state transition matrix based off the a-priori state F(x_(k/k-1))
        - The process noise matrix Q_k
    Therefore, all input lists should be the same length

    The smoothed state and covarinace is described by
    C_k = P_(k/k) F(k+1).T P(k+1/k)^-1
    x_(k/n) = x_(k/k) + C_k*(x_(k+1/n) - x_(k_1/k))
    P_(k/n) = P_(k/k) + C_k*(P_(k+1/n) - P_(k_1/k)) C_k.T

    Parameters
    ----------
    xs : List[np.array]
        List of 1-dimensional state vectors
    Ps : List[np.array]
        List of 2-dimension covariance matrices
    x_prs : [TODO:type]
        List of 1-dimensional a-priori state vectors
    P_prs : [TODO:type]
        List of 2-dimensional a-priori covariance matrices
    Fs : [TODO:type]
        List of 2-dimensional state transition matrices
    Qs : [TODO:type]
        List of d-dimensional process noise matrices
    """

    n = len(xs)
    if not all(n_i == n for n_i in [len(Ps), len(Fs), len(Qs), len(x_prs), len(P_prs)]):
        print("ERROR: RTS smoother inputs are not same length")

    x_rts = [None]*n
    P_rts = [None]*n
    x_rts[-1] = xs[-1]
    P_rts[-1] = Ps[-1]
    for k in range(n-2, -1, -1):
        C = Ps[k]@(Fs[k+1].T)@np.linalg.inv(P_prs[k+1])
        x_rts[k] = xs[k] + C@(x_rts[k+1] - x_prs[k+1])
        P_rts[k] = Ps[k] + C@(P_rts[k+1] - P_prs[k+1])@(C.T)
    return x_rts, P_rts
