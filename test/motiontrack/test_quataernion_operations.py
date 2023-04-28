""" Test quaternion functions
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from motiontrack.utils import euler_to_quaternion, quaternion_to_euler, \
    quaternion_subtract, quaternion_multiply, quaternion_weighted_av

# Note: scipy uses scalar-last quaternion notation, whereas we use
# rotation-first
def test_quaternion_operations():

    e0 = np.array([0.1, 0.2, 0.3])
    e1 = np.array([0.2, 0.2, 0.3])
    e2 = np.array([0.5, 0.2, 0.3])
    q0 = np.array([0.9833474, 0.1435722, 0.1060205, 0.0342708])

    q0_ = euler_to_quaternion(*e0)
    np.testing.assert_almost_equal(q0_, q0)

    e0_ = quaternion_to_euler(*q0)
    np.testing.assert_almost_equal(e0_, e0)

    q1 = euler_to_quaternion(*e1)
    q_dif = quaternion_subtract(q1, q0)
    e_dif = quaternion_to_euler(*q_dif)
    np.testing.assert_almost_equal(e_dif, np.array([0.1, 0, 0]))

    q2 = (R.from_quat(q0[[1,2,3,0]])*R.from_quat(q0[[1,2,3,0]])).as_quat()
    q2_ = quaternion_multiply(q0, q0)
    np.testing.assert_almost_equal(q2_, q2[[3,0,1,2]])

    q0 = euler_to_quaternion(*e0)
    q2 = euler_to_quaternion(*e2)
    #  ws = [0.5, 0.5]
    ws = [0.75, 0.25]
    qs = [q0, q2]
    q_av = quaternion_weighted_av(qs, ws)
    e_av = quaternion_to_euler(*q_av)

if __name__ == "__main__":
    test_quaternion_operations()
