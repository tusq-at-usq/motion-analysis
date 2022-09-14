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
from matplotlib import pyplot as plt

from motiontrack.sample_bodies.cube import make_cube
from motiontrack.utils import euler_to_quaternion
from motiontrack.body_projection import View

from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group

from motiontrack.plot import PlotMatch, PlotTrack

from kalmanfilter.ekf import ExtendedKalmanFilter
from kalmanfilter.process_noise import custom_process_noise

sys.path.insert(1, 'classes/')
from blob_track_observable import BlobTrack6DoF

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
    P0 = np.eye(dsys.get_nx())*1

    x0 = dsys.load_x_0(CONFIG)
    ekf = ExtendedKalmanFilter(dsys, 0.1, quaternions=True)
    ekf.initialise(x0, P=P0)
    Q_c = np.zeros((13,13))
    Q_c[7:13,7:13] = np.eye(6)

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
    x_pr_history = []
    t_history = []
    y_history = []

    t = 0
    x = x0
    t_next, ob_next_group, nz = get_next_obs_group(obs_list, t, 0.0001)
    while t_next < np.inf:
        dt = t_next - t
        # Create observable function, system Jacboian matrix, observation
        # Jacobian matrix, and process noise.
        hx = create_observable_function(ob_next_group, dsys.x_dict)
        A = dsys.J_np(x,[])
        H = ekf.create_H_jacobian(x, nz, hx, dt)
        Q = custom_process_noise(A, Q_c, dt, 500)
        breakpoint()

        # Predict
        x_pr = ekf.predict(dt, Q)

        # Update
        y, R = get_all_measurements(ob_next_group, x_pr, dsys.x_dict)
        x, P = ekf.update(y, R, hx, H)

        # store results
        x_history.append(x)
        x_pr_history.append(x_pr)
        t_history.append(t_next)
        y_history.append(y)

        # Get next measurement time and observation group
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

        body.plot()
        plot_track.update_state(x_history, t_history)
        #  plot_track.update_priori(x_pr_history, t_history)
        plot_track.update_observation(y_history, t_history)

        print('Step no:',len(t_history),'complete')
        if debug:
            print("A priori:")
            print(x_pr)
            print("Tracked:")
            print(x)
            print("True:")
            print(true_data[len(t_history)-1,:])
            print("dt = ",dt)

            input("Press key for next step")

    plot_t.close()
    plot_e.close()

    input("Press key to close figures")
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




