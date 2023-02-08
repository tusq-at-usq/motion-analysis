""" Test case for Dynamicdsysystem class
"""

import pickle
import numpy as np
import os
#  from matplotlib import pyplot as plt

CONFIG = os.path.dirname(__file__)+'/data/projectile_config'
FILE  = os.path.dirname(__file__)+'/data/projectile_system'

def test_dynamic_system():

    with open(FILE,'rb') as file:
        dsys = pickle.load(file)

    dsys.write_default_config('test')
    os.remove("test.yaml")
    dsys.load_config(CONFIG)
    dsys.add_aug_states(CONFIG)
    dsys.sub_params(CONFIG)
    dsys.J = dsys._create_jacobian()
    dsys.lambdify()

    x_0 = dsys.load_x_0(CONFIG)
    x_history, t_history = dsys.integrate(2.8,x_0,dt_max=0.1)
    x_last = np.array([24.30034925, 5.66086674,
                       1.05025819, 4.51467575,
                      -4.91091348, -23.02609882])

    np.testing.assert_array_almost_equal(x_history[-1],x_last)

if __name__=="__main__":
    test_dynamic_system()
