
import pickle
import numpy as np
from matplotlib import pyplot as plt

CONFIG = '../data/system_config'
FILE  = '../data/six_dof_quaternions'

with open(FILE,'rb') as file:
    dsys = pickle.load(file)

#  dsys.write_default_config(CONFIG)

dsys.load_config(CONFIG)
dsys.add_aug_states(CONFIG)
dsys.sub_params(CONFIG)
dsys.J = dsys._create_jacobian()
dsys.lambdify()
x_0 = dsys.load_x_0(CONFIG)
x_history_1, t_history_1 = dsys.integrate(2.5,x_0,dt_max=0.02)
x_history_2, t_history_2 = dsys.integrate(2.5,x_0,dt_max=0.05)

POS_E_SCALE = 0.5
Q_E_SCALE = 0.2

error1 = np.concatenate([np.random.normal(0,POS_E_SCALE,[len(t_history_1),2]),
                        np.random.normal(0,Q_E_SCALE,[len(t_history_1),4])], axis=1)
error2 = np.concatenate([np.random.normal(0,POS_E_SCALE,[len(t_history_2),2]),
                        np.random.normal(0,Q_E_SCALE,[len(t_history_2),4])], axis=1)

x_data_error_1 = x_history_1[:,[0,2,3,4,5,6]]+error1
x_data_error_2 = x_history_2[:,[1,2,3,4,5,6]]+error2

# Extra step to normalise the quaternions after error
for i,_ in enumerate(t_history_1):
    x_data_error_1[i,2:6] = x_data_error_1[i,2:6]/np.linalg.norm(x_data_error_1[i,2:6])
for i,_ in enumerate(t_history_2):
    x_data_error_2[i,2:6] = x_data_error_2[i,2:6]/np.linalg.norm(x_data_error_2[i,2:6])

np.savetxt("../data/view1_2ob.csv", x_data_error_1, delimiter=",")
np.savetxt("../data/view2_2ob.csv", x_data_error_2, delimiter=",")
np.savetxt("../data/time_1_2ob.csv", t_history_1, delimiter=",")
np.savetxt("../data/time_2_2ob.csv", t_history_2, delimiter=",")
np.savetxt("../data/true_2ob.csv", x_history_1, delimiter=",")
np.savetxt("../data/time_true_2ob.csv", t_history_1, delimiter=",")

ax = plt.axes(projection='3d')
ax.plot(x_history_1[:,0], x_history_1[:,1], x_history_1[:,2],
        color='k', label="true")
plt.show()
