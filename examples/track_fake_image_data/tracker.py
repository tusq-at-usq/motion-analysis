# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments

""" Main tracking loop

Pre:
    - Initialise observables (viewpoints, geometry etc.)
    - Initilialise dynamic system
    - Initilaise any observable classes
    - Get intiial state vector
    - Initialise Kalman filter

Loop:
    1.  Get next timestep, and sub-list of observable groups
    2.  Create observable function h(x), uncertanty matrix R, and process
        uncertainty matrix Q
    3.  Get x_pr and z_pr a priori estimates
    4.  Get measurements y
    3.  Update Kalman filter
    4.  Store timestep results
"""

import pickle
import sys
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt

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

from filterpy.kalman import KalmanFilter as EKF

CONFIG = 'data/system_config_errors'
#  CONFIG = 'data/system_config'
SYSTEM  = 'data/six_dof_quaternions'
BLOB_FILES = ['data/view_1_data.csv', 'data/view_2_data.csv']
TRUE_DATA = 'data/true.csv'
TRUE_TIME = 'data/time.csv'


def test_tracking_loop(print_flag=0, debug=False):

    # ---------------------------------------------------------------
    # Create Body and View objects
    # ---------------------------------------------------------------
    body = make_cube()
    Q = euler_to_quaternion(0, 0, 0)
    body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

    view_t = View(body, np.array([-np.pi/2,0.00000,0.000000]), "top", "top")
    view_e = View(body, np.array([0, 0, np.pi/2]), "east", "east")


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
    #  P0 = np.ones((dsys.get_nx(), dsys.get_nx()))*10
    P0 = np.eye(dsys.get_nx())*1000
    #  P0 = np.ones((dsys.get_nx(), dsys.get_nx()))*100 + np.eye(dsys.get_nx())*1000

    x0 = dsys.load_x_0(CONFIG)
    ekf = ExtendedKalmanFilter(dsys, 0.1, quaternions=False)
    ekf.initialise(x0, P=P0)
    Q_c = np.ones((13,13))*0.01
    Q_c[7:13,7:13] = np.eye(6)

    fp = EKF(13,7)
    fp.x = x0
    fp.P = P0

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

    x_fp_history = []
    P_fp_history = []

    t = 0
    x = x0
    P = P0
    t_next, ob_next_group, nz = get_next_obs_group(obs_list, t, 0.0001)
    y, R = get_all_measurements(ob_next_group, x0, dsys.x_dict)
    t_next, ob_next_group, nz = get_next_obs_group(obs_list, t, 0.0001)
    while t_next < np.inf:
        dt = t_next - t
        # Create observable function, system Jacboian matrix, observation
        # Jacobian matrix, and process noise.
        hx = create_observable_function(ob_next_group, dsys.x_dict)
        A = dsys.J_np(x,[])
        Q = custom_process_noise(A, Q_c, dt, 200)
        #  Q= np.eye(13)*10

        # Predict
        x_priori, P_priori, F_priori = ekf.predict(dt, Q)
        #TODO - get a-priori covariance estimate
        #  x_pr = F@x

        #  print(x_pr1)
        #  print(x_pr)
        H = ekf.create_H_jacobian(x_priori, nz, hx, dt)

        # Get measurements
        y, R = get_all_measurements(ob_next_group, x_priori, dsys.x_dict)

        # Store results
        t_history.append(t_next)
        y_history.append(y)
        x_pr_history.append(x_priori)
        P_pr_history.append(P_priori)

        # Update
        x, P = ekf.update(y, R, hx, H)
        x_history.append(x.copy())
        P_history.append(P.copy())
        Q_history.append(Q)
        F_history.append(F_priori)

        def Hjacobian(x_):
            return ekf.create_H_jacobian(x_,nz,hx,dt)

        fp.Q = Q
        fp.F = F_priori
        fp.R = R
        fp.H = H

        # Filterpy predict
        def predict_x(u):
            return np.array(x_pr).copy()

        fp.predict_x = predict_x

        fp.predict()
        # Filterpy update
        fp.update(y, R, H)

        x_fp_history.append(fp.x)
        P_fp_history.append(fp.P)

        # Get next measurement time and observation group
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

        body.plot()
        plot_track.update_state(x_history, t_history)
        #  plot_track.update_priori(x_pr_history, t_history)
        plot_track.update_observation(y_history, t_history)
        #  input("Press to update")
        #  plot_track.update_state(x_fp_history, t_history)

        print('Step no:',len(t_history),'complete')
        if debug:
            print("A priori:")
            print(x_priori)
            print("Tracked:")
            print(x)
            print("True:")
            print(true_data[len(t_history)-1,:])
            print("dt = ",dt)
            print("P = ")
            print(P)
            input("Press key for next step")

    plot_t.close()
    plot_e.close()

    x_rts, P_rts = rts_smoother(x_history,
                                P_history,
                                x_pr_history,
                                P_pr_history,
                                F_history,
                                Q_history)

    #  breakpoint()

    #  from filterpy.kalman import KalmanFilter
    #  kf = KalmanFilter(13,7)
    #  x_rts, P_rts, _, _ = fp.rts_smoother(np.array(x_history),
                               #  np.array(P_history),
                               #  np.array(F_history),
                               #  np.array(Q_history))

    plot_track.load_smoothed_data(x_rts, t_history)

    #  plot_track.load_smoothed_data(x_rts, t_history)

    breakpoint()
    #  input("Press key to close figures")
    plot_track.close()

    x_history = np.array(x_history)
    x_pr_history = np.array(x_pr_history)
    t_history = np.array(t_history)

    #  ---------------------------------------------------------------------
    #  Plot results
    #  ---------------------------------------------------------------------

    #  XYZ plot

    #  ax = plt.axes(projection='3d')
    #  ax.plot(true_data[:,0], true_data[:,1], true_data[:,2],
            #  color='k', label="true")
    #  ax.scatter3D(x_history[:,0], x_history[:,1], x_history[:,2], label='track')
    #  ax.set_xlabel('x')
    #  ax.set_ylabel('y')
    #  ax.set_zlabel('z')
    #  ax.legend()
    #  ax.set_title("State-space")
    #  plt.show()

    #  Observations plot

    #  hx_1 = obs_1.create_ob_fn(dsys.x_dict)
    #  y1_true = np.array([hx_1(x_t) for x_t in true_data])

    #  y1_tracked = np.array([hx_1(x_t) for x_t in x_history])

    #  y1 = np.array(obs_1.y_history)

    #  fig, ax = plt.subplots(1,1)
    #  ax = ax.flatten()

    #  ax.plot(obs_1.t_history, y1[:,3:7],'o')
    #  ax.plot(true_time, y1_true[:,3:7],'-',color='k')
    #  ax.plot(t_history, y1_tracked[:,3:7],'-',color='r')
    #  ax.set_title('Quaternions')

    #  ax[0].plot(obs_2.t_history, y2,'o')
    #  ax[0].plot(true_time, y1_true[:,3:7],'-',color='k')
    #  ax[0].plot(t_history, y1_tracked[:,3:7],'-',color='r')
    #  ax[0].set_title('Observable 2')

    #  plt.show()

if __name__=='__main__':
    test_tracking_loop(1, debug=False)




