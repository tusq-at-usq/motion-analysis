""" Test case for Dynamicdsysystem class
"""

import pickle
from matplotlib import pyplot as plt

CONFIG = 'data/projectile_config'
FILE  = 'data/projectile_system'

with open(FILE,'rb') as file:
    dsys = pickle.load(file)

#  dsys.write_default_config('test')
dsys.load_config(CONFIG)
dsys.add_aug_states(CONFIG)
dsys.sub_params(CONFIG)
dsys.lambdify()
x_0 = dsys.load_x_0(CONFIG)
x_history, t_history = dsys.integrate(2.8,x_0,dt_max=0.1)
print("Integrated state history:", x_history)
print("Time history: ", t_history)

ax = plt.axes(projection='3d')
ax.plot(x_history[:,0], x_history[:,1], x_history[:,2],color='k')
ax.scatter3D(x_history[:,0], x_history[:,1], x_history[:,2], c=t_history)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


