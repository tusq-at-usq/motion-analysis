""" Test combining multiple obseravbles into one measurement
"""

import numpy as np
import sys
import pickle
from motiontrack.combine_observables import get_all_measurements,\
    create_observable_function, get_next_obs_group

sys.path.insert(1, './classes/')
from position_observable import Position2D

CONFIG = 'data/projectile_config'
FILE  = 'data/projectile_system'

with open(FILE,'rb') as file:
    dsys = pickle.load(file)


pos_2d_1 = Position2D(name='view1',
                      ob_names=['x','z'],
                      t_filename='data/time.csv',
                      data_filename='data/view1.csv')
pos_2d_2 = Position2D(name='view2',
                      ob_names=['y','z'],
                      t_filename='data/time.csv',
                      data_filename='data/view2.csv')

obs_list = [pos_2d_1, pos_2d_2]


x_prior = np.zeros(6)

t, ob_next_group, nz = get_next_obs_group(obs_list,-1,0.0001)
while t < np.inf:
    hx = create_observable_function(ob_next_group, dsys.x_dict)
    y, tau = get_all_measurements(obs_list, x_prior,dsys.x_dict)
    t, ob_next_group, nz = get_next_obs_group(obs_list,-1,0.0001)
    print(y)

