# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments

""" Main tracking loop
Pre:
    - Initilialise dynamic system
    - Initilaise any observable classes
    - Get intiial state vector
    - Initialise Kalman filter

Loop:
    1.  Get next timestep, and sub-list of observable groups
    2.  Create observable function h(x), uncertanty matrix R, system Jacobian A,
    observation Jacobian H, and and process uncertainty matrix Q
    3.  Get a priori estimates (x_pr)
    4.  Get measurements (y)
    3.  Update Kalman filter
    4.  Store timestep results
"""

import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt

from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group

from kalmanfilter.ekf import ExtendedKalmanFilter
from kalmanfilter.process_noise import custom_process_noise

sys.path.insert(1, 'classes/')
from position_observable import Pos2DandQ

CONFIG = 'data/system_config_error'
SYSTEM  = 'data/six_dof_quaternions'
TRUE_DATA = 'data/true_2ob.csv'
TRUE_TIME = 'data/time_true_2ob.csv'

def test_tracking_loop(print_flag=0):
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

    # ---------------------------------------------------------------------
    # Define observable groups (i.e. two camera views)
    # ---------------------------------------------------------------------
    pos_2d_1 = Pos2DandQ(name='view1',
                          ob_names=['x','z','q0','q1','q2','q3'],
                          t_filename='data/time_1_2ob.csv',
                          data_filename='data/view1_2ob.csv')
    pos_2d_2 = Pos2DandQ(name='view2',
                          ob_names=['y','z','q0','q1','q2','q3'],
                          t_filename='data/time_2_2ob.csv',
                          data_filename='data/view2_2ob.csv')
    obs_list = [pos_2d_1, pos_2d_2]

    # ---------------------------------------------------------------------
    # Define filter
    # ---------------------------------------------------------------------
    P0 = np.eye(dsys.get_nx())*10000
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
        Q = custom_process_noise(A, Q_c, dt, 1)

        # Predict
        x_pr = ekf.predict(dt, Q)

        # Update
        y, R = get_all_measurements(ob_next_group, x_pr, dsys.x_dict)
        x, P = ekf.update(y, R, hx, H)

        # store results
        x_history.append(x)
        x_pr_history.append(x_pr)
        t_history.append(t)
        y_history.append(y)

        # Get next measurement time and observation group
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

    x_history = np.array(x_history)
    x_pr_history = np.array(x_pr_history)
    t_history = np.array(t_history)

    # ---------------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(5,3, figsize=(10,12))
    ax = ax.flatten()

    # States - tracked vs true
    labels = ["x","y","z","q0", "q1", "q2", "q3","u","v","w","p","q","r"]
    for i in range(13):
        ax[i].plot(true_time, true_data[:,i],'--', color='k', linewidth=1, label='True')
        ax[i].plot(t_history, x_history[:,i],'r', linewidth=1, label='Tracked')
        ax[i].plot(t_history, x_pr_history[:,i],'b', linewidth=1, label='Priori')
        ax[i].set_title(labels[i])
    ax[0].legend()
    plt.tight_layout()
    plt.show()

    # Obsevations - data vs true vs tracked
    hx_1 = pos_2d_1.create_ob_fn(dsys.x_dict)
    hx_2 = pos_2d_2.create_ob_fn(dsys.x_dict)
    y1_true = np.array([hx_1(x_t) for x_t in true_data])
    y2_true = np.array([hx_2(x_t) for x_t in true_data])
    y1_tracked = np.array([hx_1(x_t) for x_t in x_history])
    y2_tracked = np.array([hx_2(x_t) for x_t in x_history])
    y1 = np.array(pos_2d_1.y_history)
    y2 = np.array(pos_2d_2.y_history)

    fig, ax = plt.subplots(2,1)
    ax = ax.flatten()

    ax[0].plot(pos_2d_1.t_history, y1,'.')
    ax[0].plot(pos_2d_1.t_history, y1_true,'-',color='k')
    ax[0].plot(t_history, y1_tracked,'-',color='r',linewidth=0.75)
    ax[0].set_title('Observable 1')

    ax[1].plot(pos_2d_2.t_history, y2,'.')
    ax[1].plot(pos_2d_1.t_history, y2_true,'-',color='k')
    ax[1].plot(t_history, y2_tracked,'-',color='r',linewidth=0.75)
    ax[1].set_title('Observable 2')

    plt.show()

if __name__=='__main__':
    test_tracking_loop(1)




