
import pickle
import numpy as np

CONFIG = '../data/projectile_config'
FILE  = '../data/projectile_system'

with open(FILE,'rb') as file:
    dsys = pickle.load(file)

dsys.load_config(CONFIG)
dsys.add_aug_states(CONFIG)
dsys.sub_params(CONFIG)
dsys.lambdify()
x_0 = dsys.load_x_0(CONFIG)
x_history, t_history = dsys.integrate(2.8,x_0,dt_max=0.05)

error1 = np.random.normal(0,1,[len(t_history),2])
error2 = np.random.normal(0,1,[len(t_history),2])

np.savetxt("../data/view1.csv", x_history[:,[0,2]]+error1, delimiter=",")
np.savetxt("../data/view2.csv", x_history[:,[1,2]]+error2, delimiter=",")
np.savetxt("../data/time.csv", t_history, delimiter=",")

np.savetxt("../data/true.csv", x_history, delimiter=",")
