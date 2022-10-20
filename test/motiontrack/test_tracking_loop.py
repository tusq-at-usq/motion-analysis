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

from filterpy.common import Q_continuous_white_noise
from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group
from kalmanfilter.ekf import ExtendedKalmanFilter

sys.path.insert(1, 'classes/')
from position_observable import Position2D

CONFIG = 'data/projectile_config'
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
    P0 = np.eye(dsys.get_nx())*2
    x0 = dsys.load_x_0(CONFIG)
    ukf = ExtendedKalmanFilter(dsys, 0.1)
    ukf.initialise(x0, P=P0)

    # ---------------------------------------------------------------------
    # Load true data for compairson
    # ---------------------------------------------------------------------

    true_data = np.loadtxt(TRUE_DATA, delimiter=',')

    # ---------------------------------------------------------------------
    # Create main loop
    # ---------------------------------------------------------------------

    x_history = []
    t_history = []
    y_history = []

    i=0
    t = 0
    t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)
    while t_next < np.inf:
        #  breakpoint()
        dt = t_next - t
        Q = Q_continuous_white_noise(2, dt, 1, 3, order_by_dim=False)
        hx = create_observable_function(ob_next_group, dsys.x_dict)
        x_pr, z_pr = ukf.predict(dt, nz, hx, Q)

        y, R = get_all_measurements(obs_list, x_pr, dsys.x_dict)
        x, P = ukf.update(y, R)
        t = t_next
        t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.0001)

        x_history.append(x)
        t_history.append(t)
        y_history.append(y)

    x_history = np.array(x_history)
    t_history = np.array(t_history)


    ax = plt.axes(projection='3d')
    ax.plot(true_data[:,0], true_data[:,1], true_data[:,2],
            color='k', label="true")
    #  ax.plot(x_history[:,0], x_history[:,1], x_history[:,2],
            #  color='r', label="track")
    ax.scatter3D(x_history[:,0], x_history[:,1], x_history[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title("State-space")
    plt.draw()
    plt.pause(1)
    plt.close()

if __name__=='__main__':
    test_tracking_loop(1)




