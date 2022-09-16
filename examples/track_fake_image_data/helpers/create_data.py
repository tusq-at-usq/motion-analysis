""" Create geometry and views needed to create fake data and track
"""
import pickle
import numpy as np

from motiontrack.sample_bodies.cube import make_cube
from motiontrack.body_projection import View
from motiontrack.blob_data import write_blob_data
from motiontrack.utils import euler_to_quaternion

CONFIG = '../data/system_config'
FILE  = '../data/six_dof_quaternions'

# ---------------------------------------------------------------
# Create Body and View objects
# ---------------------------------------------------------------
body = make_cube()
Q = euler_to_quaternion(0, 0, 0)
body.initialise([0,0,0], [Q[0], Q[1], Q[2], Q[3]])

view_1 = View(body, np.array([-np.pi/2,0.00000,0.000000]), "top", "top")
view_2 = View(body, np.array([0, 0, np.pi/2]), "east", "east")

# ---------------------------------------------------------------
# Create dynamic system data
# ---------------------------------------------------------------
with open(FILE,'rb') as file:
    dsys = pickle.load(file)
#  dsys.write_default_config(CONFIG)
dsys.load_config(CONFIG)
dsys.add_aug_states(CONFIG)
dsys.sub_params(CONFIG)
dsys.lambdify()
x_0 = dsys.load_x_0(CONFIG)
x_history, t_history = dsys.integrate(2,x_0,dt_max=0.02)
pos_history = x_history[:,0:3]
q_history = x_history[:,3:7]

np.savetxt("../data/true.csv", x_history, delimiter=",")
np.savetxt("../data/time.csv", t_history, delimiter=",")

# ---------------------------------------------------------------
# Loop over history, updating and storing the blob locations
# ---------------------------------------------------------------
ERROR_SCALE = 2
blob_locs_1 = {}
blob_locs_2 = {}
for i,(pos, q) in enumerate(zip(pos_history, q_history)):
    body.update(pos*1000,q)
    blob_locs_1[i] = view_1.get_blobs()
    blob_locs_2[i] = view_2.get_blobs()
write_blob_data(blob_locs_1, "../data/view_1_data.csv", ERROR_SCALE)
write_blob_data(blob_locs_2, "../data/view_2_data.csv", ERROR_SCALE)






