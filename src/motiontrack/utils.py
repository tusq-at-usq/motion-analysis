""" A selection of utility functions, including quaternion operations
"""

from typing import List, Union
import numpy as np
import sympy as sp

from scipy.spatial.transform import Rotation as R

def wxyz_to_quaternion(w_rad, x, y, z):
    q0 = np.cos(w_rad/2)     
    q1 = np.sin(w_rad/2)*x
    q2 = np.sin(w_rad/2)*y
    q3 = np.sin(w_rad/2)*z
    return np.array([q0, q1, q2, q3])

def quaternion_to_rotation_tensor(q0: float, q1: float, q2: float, q3: float,
                       package: str='numpy') -> Union[np.array, sp.Matrix]:
    """
    Quaternion to rotation tensor (Eqn 10.14 from Zipfel)

    Parameters
    ----------
    q0 : float
        Quaternion scalar component
    q1 : float
        quaternion rotation vector i
    q2 : float
        quatenrion rotation vector j
    q3 : float
        quaternion rotation vector k
    package: str, default = 'numpy'
        data type to use for return
    """
    if package == 'numpy':
        T_rot = np.array([
            [q0**2+q1**2-q2**2-q3**2,  2*(q1*q2+q0*q3),  2*(q1*q3-q0*q2) ],
            [2*(q1*q2-q0*q3),  q0**2-q1**2+q2**2-q3**2,  2*(q2*q3+q0*q1) ],
            [2*(q1*q3+q0*q2),  2*(q2*q3-q0*q1),  q0**2-q1**2-q2**2+q3**2 ]
            ]).T # Eqn 10.14 from Zipfel
    elif package == 'sympy':
        T_rot = sp.Matrix([
            [q0**2+q1**2-q2**2-q3**2,  2*(q1*q2+q0*q3),  2*(q1*q3-q0*q2) ],
            [2*(q1*q2-q0*q3),  q0**2-q1**2+q2**2-q3**2,  2*(q2*q3+q0*q1) ],
            [2*(q1*q3+q0*q2),  2*(q2*q3-q0*q1),  q0**2-q1**2-q2**2+q3**2 ]
            ]).T
    else:
        return ValueError("Package type not recognised")
    return T_rot

def quaternion_to_rotation_tensor_SP(q0: float, q1: float, q2: float, q3: float)\
        -> np.array:
    """
    Quaternion to rotation tensor, using SciPy package

    Function quaternion input order is a + bi + cj + dk, but note that 
    the SciPy package orders them differently, with the scalar component last.

    Parameters
    ----------
    q0 : float
        Quaternion scalar rotation component
    q1 : float
        Quaternion i vector component
    q2 : float
        Quaternion j vector component
    q3 : float
        Quaternion k vector component

    Returns
    -------
    T_rot: np.array
        Rotation matrix (3x3)
    """
    T_rot = R.from_quat([q1, q2, q3, q0]).as_matrix()
    return T_rot

def quaternion_to_matrix(q: Union[np.array, List[float]]) -> np.array:
    """
    Matrix representation of quaternion

    Quaternion order is q = {a1 + bi + cj + dk}

    Parameters
    ----------
    q : ndarray or list
        Quaternion vector to convert to matrix representation
    """
    Q = np.array([[q[0], q[1], q[2], q[3]],
                  [-q[1], q[0], q[3], -q[2]],
                  [-q[2], -q[3], q[0], q[1]],
                  [-q[3], q[2], -q[1], q[0]]])
    return Q

def quaternion_multiply(q_mul: np.array, q0: np.array) -> np.array:
    """
    Quaternion multiplication.

    Note multiplication is not commutative. 
    Quaternion order is q = {a1 + bi + cj + k}.
    Operation by converting both to matrix form, then using dot product

    Parameters
    ----------
    q_mul : np.array
        Quaternion for multiplication operation (i.e. rotation quaternion)
    q0 : np.array
        Original quaternion, to apply multiplication to

    Returns
    -------
    Q2: np.array
        Resultant quaternion
    """
    Q0 = quaternion_to_matrix(q0)
    Q_mul = quaternion_to_matrix(q_mul)
    Q2 = Q_mul@Q0
    q2 = Q2[0,:]
    q2 = q2/np.linalg.norm(q2)
    return q2

def quaternion_subtract(q_from: np.array, q_to: np.array) -> np.array:
    """
    Quaternion subtraction

    Note multiplication is not commutative.
    Quaternion order is q = {a1 + bi + cj + dk}.
    Operation by converting both to matrix form, then using dot product of
    quaternion matrix inverse.

    Parameters
    ----------
    q0 : np.array
        Original quaternion
    q_sub : np.array
        Quaternion to subtract from q0

    Returns
    -------
    Q_diff: np.array
        Resultant quaternion
    """
    Q_from = quaternion_to_matrix(q_from)
    Q_to = quaternion_to_matrix(q_to)
    Q_diff = np.linalg.inv(Q_from) @ Q_to
    q_diff = Q_diff[:,0]
    q_diff = q_diff/np.linalg.norm(q_diff)
    return q_diff

#  def euler_to_quaternion(psi: float, theta: float, phi: float) -> List[float]:

def euler_to_quaternion(psi: float, theta: float, phi: float) -> List[float]:
    """
    Euler angles to quaternion

    According to Zipfel Eqn. 10.12
    Quaternion order is q = {a1 + bi + cj + k}.
    Euler angles are in radians, and in Tait-Bryan angles (z-y-x)

    Parameters
    ----------
    psi : float
        psi rotataion vector (radians)
    theta : float
        theta rotation vector (radians)
    phi : float
        phi rotation vector (radians)

    Returns
    -------
    q0, q1, q2, q3 : List[float]
        Equivalent quaternion
    """
    q0 = np.cos(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
        + np.sin(psi/2) * np.sin(theta/2) * np.sin(phi/2)
    q1 = np.cos(psi/2) * np.cos(theta/2) * np.sin(phi/2) \
        - np.sin(psi/2) * np.sin(theta/2) * np.cos(phi/2)
    q2 = np.cos(psi/2) * np.sin(theta/2) * np.cos(phi/2) \
        + np.sin(psi/2) * np.cos(theta/2) * np.sin(phi/2)
    q3 = np.sin(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
        - np.cos(psi/2) * np.sin(theta/2) * np.sin(phi/2)
    Q = np.array([q0, q1, q2, q3])
    Q = Q/np.linalg.norm(Q)
    return Q

def quaternion_to_euler(q0: float, q1: float, q2: float, q3: float) \
        -> List[float]:
    """
    Quaternions to Euler angles

    # According to Zipfel Eqn. 10.12
    Quaternion order is q = {a1 + bi + cj + k}.
    Euler angles are in radians, and in Tait-Bryan angles (z-y-x)

    Parameters
    ----------
    Parameters
    ----------
    q0 : float
        Quaternion scalar component
    q1 : float
        quaternion rotation vector i
    q2 : float
        quatenrion rotation vector j
    q3 : float
        quaternion rotation vector k

    Returns
    -------
    psi, theta, phi : List[float]
        Tait-Bryan Euler angles
    """
    psi = np.arctan2(2 * (q1 * q2 + q0 * q3),
           (q0**2 + q1**2 - q2**2 - q3**2))
    theta = np.arcsin(-2 * (q1 * q3 - q0 * q2))
    phi = np.arctan2(2 * (q2 * q3 + q0 * q1),
           (q0**2 - q1**2 - q2**2 + q3**2))
    # TODO: Need to adjust code to manage singularities at psi = +/- pi/2 and phi = +/- pi/2
    return psi, theta, phi

def euler_to_rotation_tensor(psi: float, theta: float, phi: float) -> np.array:
    Q = euler_to_quaternion(psi, theta, phi)
    R = quaternion_to_rotation_tensor(*Q)
    return R

def quaternion_weighted_av(qs, ws):
    #  qw = [q*(w**0.5) for q,w in zip(qs,ws)]
    QW = np.array(qs).T
    W = np.eye(len(ws))*ws
    A = QW@W@(QW.T)
    eigvals, eigvecs = np.linalg.eig(A)
    i = np.argmax(eigvals)
    q_i = eigvecs[:,i]
    q_av = q_i/np.linalg.norm(q_i)
    return q_av

def tensor_to_quaternion(tensor):
    Q = R.from_matrix(tensor).as_quat()
    Q = np.array([Q[3], Q[0], Q[1], Q[2]])
    return Q
