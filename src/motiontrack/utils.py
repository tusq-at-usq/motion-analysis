import numpy as np

from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(psi, theta, phi):
    """Convert Euler angles to quaternions."""
    # According to Zipfel Eqn. 10.12
    q0 = np.cos(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
        + np.sin(psi/2) * np.sin(theta/2) * np.sin(phi/2)
    q1 = np.cos(psi/2) * np.cos(theta/2) * np.sin(phi/2) \
        - np.sin(psi/2) * np.sin(theta/2) * np.cos(phi/2)
    q2 = np.cos(psi/2) * np.sin(theta/2) * np.cos(phi/2) \
        + np.sin(psi/2) * np.cos(theta/2) * np.sin(phi/2)
    q3 = np.sin(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
        - np.cos(psi/2) * np.sin(theta/2) * np.sin(phi/2)
    return q0, q1, q2, q3

def quaternion_to_euler(q0, q1, q2, q3):
    """Convert Quaternion to Euler angles."""
    # According to Zipfel Eqn. 10.12
    psi = np.arctan(2 * (q1 * q2 + q0 * q3)
          / (q0**2 + q1**2 - q2**2 - q3**2))
    theta = np.arcsin(-2 * (q1 * q3 - q0 * q2))
    phi = np.arctan(2 * (q2 * q3 + q0 * q1)
          / (q0**2 - q1**2 - q2**2 + q3**2))
    # TODO: Need to adjust code to manage singularities at psi = +/- pi/2 and phi = +/- pi/2
    return psi, theta, phi

def quaternions_to_rotation_tensor(q0, q1, q2, q3):
    """Calculate directional cosine matrix from quaternions."""
    T_rot = R.from_quat([q1, q2, q3, q0]).as_matrix()
    return T_rot

def _to_matrix(q):
    """ Quaternion order is q = {a1 + bj + cw + d} """
    Q = np.array([[q[0], q[1], q[2], q[3]],
                  [-q[1], q[0], -q[3], q[2]],
                  [-q[2], q[3], q[0], -q[1]],
                  [-q[3], -q[2], q[1], q[0]]])
    return Q

def quaternion_multiply(q_mul, q0):
    """ Quaternion order is q = {a1 + bj + cw + d} """
    Q0 = _to_matrix(q0)
    Q_mul = _to_matrix(q_mul)
    Q2 = Q_mul@Q0
    return Q2[0,:]

def quaternion_subtract(q0,q_sub):
    """ Finds the quaternion rotation difference such that:
    q_diff * q_sub = q0, i.e. multipl q0 by dq
    """
    Q0 = _to_matrix(q0)
    Q_sub = _to_matrix(q_sub)
    Q_diff = Q0@np.linalg.inv(Q_sub)
    return Q_diff[0,:]

