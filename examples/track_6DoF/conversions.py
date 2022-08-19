
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
    """Convert Quternion to Euler angles."""
    # According to Zipfel Eqn. 10.12
    psi = np.atan2(2 * (q1 * q2 + q0 * q3)
        / (q0**2 + q1**2 - q2**2 - q3**2))
    theta = np.arcnp.sin(-2 * (q1 * q3 - q0 * q2))
    phi = np.atan2(2 * (q2 * q3 + q0 * q1)
        / (q0**2 - q1**2 - q2**2 + q3**2))
    # TODO: Need to adjust code to manage np.singularities at
    # psi = +/- pi/2 and phi = +/- pi/2
    return psi, theta, phi

print(euler_to_quaternion(0,0,0))
