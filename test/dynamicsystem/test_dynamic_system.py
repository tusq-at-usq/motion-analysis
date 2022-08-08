""" Test case for DynamicSystem class
"""

import pickle

CONFIG = 'data/test'
FILE  = 'data/DoF3'

with open(FILE,'rb') as file:
    S = pickle.load(file)

S.write_default_config(CONFIG)
S.load_config(CONFIG)
S.add_aug_states(CONFIG)
S.sub_params(CONFIG)
S.lambdify()
x_0 = S.load_x_0(CONFIG)
x_history, t_history = S.integrate(4,x_0,dt_max=0.1)
print("Integrated state history:", x_history)
print("Time history: ", t_history)
print("SUCCESS")


