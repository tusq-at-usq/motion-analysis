import numpy as np

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
    T_rot = np.array([
        [q0**2+q1**2-q2**2-q3**2,  2*(q1*q2+q0*q3),  2*(q1*q3-q0*q2) ],
        [2*(q1*q2-q0*q3),  q0**2-q1**2+q2**2-q3**2,  2*(q2*q3+q0*q1) ],
        [2*(q1*q3+q0*q2),  2*(q2*q3-q0*q1),  q0**2-q1**2-q2**2+q3**2 ]
        ]) # Eqn 10.14 from Zipfel
    # T_BL += 0.5* (np.eye(3,3) - T_BI.dot(T_BI.T)).dot(T_BI)
    return T_rot

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

