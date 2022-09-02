# pylint: disable=invalid-name,too-many-instance-attributes, too-many-arguments
""" Main tracking loop

Pre:
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

from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group

from kalmanfilter.ekf import ExtendedKalmanFilter
from kalmanfilter.process_noise import custom_process_noise

sys.path.insert(1, 'classes/')
from position_observable import Position2D

CONFIG = 'data/projectile_config_errors'
#  CONFIG = 'data/projectile_config'
SYSTEM  = 'data/projectile_system'
TRUE_DATA = 'data/true.csv'

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
    pos_2d_1 = Position2D(name='view1',
                          ob_names=['x','z'],
                          t_filename='data/time.csv',
                          data_filename='data/view1.csv')
    pos_2d_2 = Position2D(name='view2',
                          ob_names=['y','z'],
                          t_filename='data/time.csv',
                          data_filename='data/view2.csv')
    obs_list = [pos_2d_1, pos_2d_2]

    # ---------------------------------------------------------------------
    # Define filter
    # ---------------------------------------------------------------------
    P0 = np.eye(dsys.get_nx())*100
    x0 = dsys.load_x_0(CONFIG)
    ekf = ExtendedKalmanFilter(dsys, 0.1)
    ekf.initialise(x0, P=P0)
    Q_c = np.zeros((6,6))
    Q_c[3:6,3:6] = np.eye(3)

    # ---------------------------------------------------------------------
    # Load true data for compairson
    # ---------------------------------------------------------------------
    true_data = np.loadtxt(TRUE_DATA, delimiter=',')

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
        hx = create_observable_function(ob_next_group, dsys.x_dict)
        dt = t_next - t
        A = dsys.J_np(x,[])
        H = ekf.create_H_jacobian(x, nz, hx, dt)
        Q = custom_process_noise(A, Q_c, dt, 3).T
        #  Q = np.eye(6)*10

        x_pr = ekf.predict(dt, Q)

        y, R = get_all_measurements(ob_next_group, x_pr, dsys.x_dict)
        x, P = ekf.update(y, R, hx, H)
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

        x_history.append(x)
        x_pr_history.append(x_pr)
        t_history.append(t)
        y_history.append(y)
        print("True data:")
        print(true_data[len(x_history)-1])
        print("x_prior:")
        print(x_pr)
        print("x:")
        print(x)
        print("P:")
        print(P)

    x_history = np.array(x_history)
    x_pr_history = np.array(x_pr_history)
    t_history = np.array(t_history)

    # ---------------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(6,1, figsize=(8,8))
    ax = ax.flatten()

    labels = ["x","y","z","u","v","w"]
    for i in range(6):
        ax[i].plot(t_history, true_data[:,i],'--', color='k', linewidth=1, label='True')
        ax[i].plot(t_history, x_history[:,i],'r', linewidth=1, label='Tracked')
        ax[i].plot(t_history, x_pr_history[:,i],'b', linewidth=1, label='Priori')
        ax[i].set_title(labels[i])
    ax[0].legend()
    plt.tight_layout()
    plt.show()

    
    #  ax = plt.axes(projection='3d')
    #  ax.plot(true_data[:,0], true_data[:,1], true_data[:,2],
            #  color='k', label="true")
    #  #  ax.plot(x_history[:,0], x_history[:,1], x_history[:,2],
            #  #  color='r', label="track")
    #  ax.scatter3D(x_history[:,0], x_history[:,1], x_history[:,2], label='track')
    #  ax.set_xlabel('x')
    #  ax.set_ylabel('y')
    #  ax.set_zlabel('z')
    #  ax.legend()
    #  ax.set_title("Position")
    #  plt.show()

    #  ax = plt.axes(projection='3d')
    #  ax.plot(true_data[:,3], true_data[:,4], true_data[:,5],
            #  color='k', label="true")
    #  ax.scatter3D(x_history[:,3], x_history[:,4], x_history[:,5], label='track')
    #  ax.set_xlabel('x')
    #  ax.set_ylabel('y')
    #  ax.set_zlabel('z')
    #  ax.legend()
    #  ax.set_title("Velocity")
    #  plt.show()

    hx_1 = pos_2d_1.create_ob_fn(dsys.x_dict)
    hx_2 = pos_2d_2.create_ob_fn(dsys.x_dict)
    y1_true = np.array([hx_1(x_t) for x_t in true_data])
    y2_true = np.array([hx_2(x_t) for x_t in true_data])

    y1_tracked = np.array([hx_1(x_t) for x_t in x_history])
    y2_tracked = np.array([hx_2(x_t) for x_t in x_history])

    y1_priori = np.array([hx_1(x_t) for x_t in x_pr_history])
    y2_priori = np.array([hx_2(x_t) for x_t in x_pr_history])

    y1 = np.array(pos_2d_1.y_history)
    y2 = np.array(pos_2d_2.y_history)

    fig, ax = plt.subplots(2,1)
    ax = ax.flatten()

    ax[0].plot(pos_2d_1.t_history, y1,'.',color='orange', label='data')
    ax[0].plot(pos_2d_1.t_history, y1_true,'-',color='k', linewidth=1, label='true')
    ax[0].plot(t_history, y1_tracked,'-',color='r', linewidth=1, label='tracked')
    ax[0].plot(t_history, y1_priori,'-',color='b', linewidth=1, label='priori')
    ax[0].set_title('Observable 1')
    ax[0].legend()

    ax[1].plot(pos_2d_2.t_history, y2,'.',color='orange', label='data')
    ax[1].plot(pos_2d_1.t_history, y2_true,'-',color='k', linewidth=1)
    ax[1].plot(t_history, y2_tracked,'-',color='r', linewidth=1)
    ax[1].plot(t_history, y2_priori,'-',color='b', linewidth=1)
    ax[1].set_title('Observable 2')
    plt.show()


if __name__=='__main__':
    test_tracking_loop(1)




