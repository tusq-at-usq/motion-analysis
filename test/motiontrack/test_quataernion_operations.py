""" Test quaternion functions
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from motiontrack.utils import euler_to_quaternion, quaternion_to_euler, \
    quaternion_subtract, quaternion_multiply

# Note: scipy uses scalar-last quaternion notation, whereas we use
# rotation-first
def test_quaternion_operations():

    e0 = np.array([0.1, 0.2, 0.3])
    q0 = np.array([0.9833474, 0.1435722, 0.1060205, 0.0342708])

    q0_ = euler_to_quaternion(*e0)
    np.testing.assert_almost_equal(q0_, q0)

    e0_ = quaternion_to_euler(*q0)
    np.testing.assert_almost_equal(e0_, e0)

    q1 = quaternion_subtract(q0, q0)
    np.testing.assert_almost_equal(q1, np.array([1, 0, 0, 0]))

    q2 = (R.from_quat(q0[[1,2,3,0]])*R.from_quat(q0[[1,2,3,0]])).as_quat()
    q2_ = quaternion_multiply(q0, q0)
    np.testing.assert_almost_equal(q2_, q2[[3,0,1,2]])


