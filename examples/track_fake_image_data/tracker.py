# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments

""" Tracking 3DoF example

Using a set of simualted blob-recognition data, the position and rotation
of a cube is tracked in 3-dimensional space.


Pre:
    - Create a set of data with guassian errors (separate module /helpers/)

Init:
    - Initialise geometry and viewpoints
    - Initilialise dynamic system
    - Initilaise figures
    - Initiliase system observables
    - Initiliase filter
    - Get intiial state vector
    - Get first set of observations
    Initialise Kalman filter

Loop:
    1.  Determine timestep
    2.  Evaluate Jacobian and system process noise
    3.  Kalman filter predict step
    4.  Create measurement function and measurement function Jacobian
    5.  Get measurements and uncertainty matrix R
    6.  Kalman filter update step
    7.  Store step results
    8.  Plot results
    9.  Get next observation groups (or return end signal)
"""

import pickle
import sys
import numpy as np

from motiontrack.sample_bodies.cube import make_cube
from motiontrack.utils import euler_to_quaternion
from motiontrack.body_projection import View

from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group

from motiontrack.plot import PlotMatch, PlotTrack

from kalmanfilter.ekf import ExtendedKalmanFilter
from kalmanfilter.process_noise import custom_process_noise
from kalmanfilter.rts_smoother import rts_smoother

sys.path.insert(1, 'classes/')
from blob_track_observable import BlobTrack6DoF

#  CONFIG = 'data/system_config_errors'
CONFIG = 'data/system_config'
SYSTEM  = 'data/six_dof_quaternions'
BLOB_FILES = ['data/view_1_data.csv', 'data/view_2_data.csv']
TRUE_DATA = 'data/true.csv'
TRUE_TIME = 'data/time.csv'

#  A = np.update_observation

def test_tracking_loop():

    # ---------------------------------------------------------------
    # Create Body and View objects
    # ---------------------------------------------------------------
    body = make_cube()
    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    view_t = View(body, np.array([-np.pi/2,0.00000,0.000000]), "top")
    view_e = View(body, np.array([0, 0, np.pi/2]), "east")

    # ---------------------------------------------------------------------
    # Define dynamic system
    # ---------------------------------------------------------------------
    with open(SYSTEM,'rb') as file:
        dsys = pickle.load(file)
    dsys.load_config(CONFIG)
    dsys.add_aug_states(CONFIG)
    dsys.sub_params(CONFIG)
    dsys.create_jacobian()
    dsys.lambdify()

    # ---------------------------------------------------------------
    # Create figures
    # ---------------------------------------------------------------

    plot_t = PlotMatch('Top')
    plot_e = PlotMatch('East')

    plot_track = PlotTrack(dsys.x_dict)

    # ---------------------------------------------------------------------
    # Define observable groups (i.e. two camera views)
    # ---------------------------------------------------------------------
    obs_1 = BlobTrack6DoF(filenames=BLOB_FILES,
                          body=body,
                          views=[view_t, view_e],
                          plots=[plot_t, plot_e],
                          frame_rate=0.02)

    obs_list = [obs_1]

    # ---------------------------------------------------------------------
    # Define filter
    # ---------------------------------------------------------------------
    P0 = np.eye(dsys.get_nx())*100
    x0 = dsys.load_x_0(CONFIG)
    Q_c = np.zeros((dsys.get_nx(),dsys.get_nx()))
    Q_c[7:13,7:13] = np.eye(6) # Process noise in continuous time

    ekf = ExtendedKalmanFilter(dsys, 0.1, quaternions=True)
    ekf.initialise(x0, P=P0)

    # ---------------------------------------------------------------------
    # Load true data for compairson
    # ---------------------------------------------------------------------
    true_data = np.loadtxt(TRUE_DATA, delimiter=',')
    true_time = np.loadtxt(TRUE_TIME, delimiter=',')
    plot_track.load_true_data(true_data, true_time)

    # ---------------------------------------------------------------------
    # Main tracking loop
    # ---------------------------------------------------------------------
    x_history = []
    P_history = []
    F_history = []
    Q_history = []
    x_pr_history = []
    P_pr_history = []
    t_history = []
    y_history = []

    t = 0
    x = x0
    P = P0
    t_next, ob_next_group, nz = get_next_obs_group(obs_list, t, 0.0001)
    while t_next < np.inf:
        dt = t_next - t

        # Create system Jacobian and process noise
        J = dsys.J_np(x,[])
        Q = custom_process_noise(J, Q_c, dt, 100)

        # Predict
        x_priori, P_priori, F_priori = ekf.predict(dt, Q)

        # Measurement function Jacobian
        hx = create_observable_function(ob_next_group, dsys.x_dict)
        H = ekf.create_H_jacobian(x_priori, nz, hx, dt) #

        # Get measurements
        y, R = get_all_measurements(ob_next_group, x_priori, dsys.x_dict)

        # Update
        x, P = ekf.update(y, R, hx, H)

        # Store results
        t_history.append(t_next)
        y_history.append(y)
        x_pr_history.append(x_priori)
        P_pr_history.append(P_priori)
        x_history.append(x.copy())
        P_history.append(P.copy())
        Q_history.append(Q)
        F_history.append(F_priori)

        # Plot results
        body.plot()
        plot_track.update_state(x_history, t_history)
        plot_track.update_observation(y_history, t_history)
        print('Step no:',len(t_history),'complete')

        # Get next measurement time and observation group
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

    plot_t.close()
    plot_e.close()

    # Use RTS smoothing on results
    x_rts, P_rts = rts_smoother(x_history,
                                P_history,
                                x_pr_history,
                                P_pr_history,
                                F_history,
                                Q_history)

    plot_track.load_smoothed_data(x_rts, t_history)
    input("Press any key to close figures")
    plot_track.close()

if __name__=='__main__':
    test_tracking_loop()




