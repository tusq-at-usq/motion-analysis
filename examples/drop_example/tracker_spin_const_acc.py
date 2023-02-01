""" Tracking example

Tracking example using:
    - Blob and line observations
    - Unscented Kalman filter
    - RTS smoothing
"""

import pickle
import numpy as np
import pandas as pd

from motiontrack.utils import euler_to_quaternion
from motiontrack.camera import CameraView
from motiontrack.observation import get_all_measurements,\
    create_observable_function, get_next_obs_group
from motiontrack.plot import PlotMatch, PlotTrack

from kalmanfilter.ukf import UnscentedKalmanFilter
from kalmanfilter.process_noise import custom_process_noise

from observations.blob_position_ob import BlobPosition
from observations.body_edge_ob import BodyEdge
from observations.frame_blob_detection import get_blobs
from geometry.cube import make_cube

from initialisation import initialise

SYSTEM = 'system/constant_acceleration'
CONFIG = 'system/const_acc_drop_2'
NAME = "drop_2"

CAM1_PREFIX = "./data/drop_2/top_cam/bd104_C002H001S000100"
CAM2_PREFIX = "./data/drop_2/east_cam/bd104_C001H001S000100"
CALIBRATION = "./observations/calibration.pkl"

REINITIALIZE = False

# ---------------------------------------------------------------
# Instantiate geometry
# ---------------------------------------------------------------

body = make_cube()
Q = euler_to_quaternion(0, 0, 0)
body.initialise([0, 0, 0], [Q[0], Q[1], Q[2], Q[3]])
body.plot()

# ---------------------------------------------------------------
# Instantiate camera views
# ---------------------------------------------------------------

with open(CALIBRATION, 'rb') as f:
    cal_e, cal_t, R_L = pickle.load(f)
R_L = np.eye(3)

view_t = CameraView(body,
                    cal_t,
                    R_L,
                    "top",
                    (1024, 1024))

view_e = CameraView(body,
                    cal_e,
                    R_L,
                    "east",
                    (1024, 1024),
                    inv=True)

# ---------------------------------------------------------------------
# Instantiate dynamic system
# ---------------------------------------------------------------------

with open(SYSTEM, 'rb') as file:
    dsys = pickle.load(file)
dsys.init(CONFIG)

# ---------------------------------------------------------------
# Create figures
# ---------------------------------------------------------------

# Camera-body overlay figure
plot_t = PlotMatch('Top', view_t.resolution)
plot_e = PlotMatch('East', view_e.resolution)

# State tracking figure
plot_track = PlotTrack(dsys.x_dict)

# ---------------------------------------------------------------------
# Define observable groups (i.e. two camera views)
# ---------------------------------------------------------------------

n_frames = 140
obs_t = BlobPosition(prefix=CAM1_PREFIX,
                     body=body,
                     view=view_t,
                     plot=plot_t,
                     frame_rate=1/1000,
                     start=80,
                     n_frames=n_frames,
                     skip=0,
                     threshold=15,
                     blob_fn = get_blobs)

obs_e = BlobPosition(prefix=CAM2_PREFIX,
                     body=body,
                     view=view_e,
                     plot=plot_e,
                     frame_rate=1/1000,
                     start=80,
                     threshold=15,
                     n_frames=n_frames,
                     skip=0,
                     flip=True,
                     blob_fn = get_blobs)

lines_e = BodyEdge(prefix=CAM2_PREFIX,
                   body=body,
                   view=view_e,
                   plot=plot_e,
                   frame_rate=1/1000,
                   start=80,
                   threshold=0.02,
                   n_frames=n_frames,
                   skip=0,
                   flip=True)

lines_t = BodyEdge(prefix=CAM1_PREFIX,
                   body=body,
                   view=view_t,
                   plot=plot_t,
                   frame_rate=1/1000,
                   start=80,
                   threshold=0.02,
                   n_frames=n_frames,
                   skip=0,
                   flip=False)

obs_list = [obs_t, obs_e, lines_e, lines_t]

# ---------------------------------------------------------------------
# Initialise position and pose states (required for video tracking)
# ---------------------------------------------------------------------

if REINITIALIZE:
    x0 = dsys.load_x_0(CONFIG)
    x0 = initialise(body,
                    [view_t, view_e],
                    [plot_t, plot_e],
                    [CAM1_PREFIX+"00"+str(obs_t.start)+".tif",
                     CAM2_PREFIX+"00"+str(obs_e.start)+".tif"],
                    X0 = x0)
    dsys.write_config(CONFIG, x0)
else:
    x0 = dsys.load_x_0(CONFIG)

# ---------------------------------------------------------------------
# Define filter
# ---------------------------------------------------------------------

# Initial state variance
P0 = np.eye(dsys.get_nx())
P0[0:3,0:3] *= 1
P0[3:7,3:7] *= 0.2
P0[7:13,7:13] *= 100
P0[13:16,13:16] *= 10000

# Continuous-time process noise
Q_c = np.zeros((dsys.get_nx(),dsys.get_nx()))
Q_c[10:13,10:13] = np.eye(3)*0.01 # Angular velocity process noise
Q_c[13:16,13:16] = np.eye(3)*10 # Linear acceleration process noise

kf = UnscentedKalmanFilter(dsys, 0.0001, quaternions=True)
kf.initialise(x0, P=P0)

# ---------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------

x_history = []
P_history = []
F_history = []
Q_history = []
t_history = []
hx_history = []
dt_history = []

t = -1e-6
x = x0
P = P0
t_next, ob_next_group, nz = get_next_obs_group(obs_list, t, 0.00001)
while t_next < np.inf:
    dt = t_next - t
    t = t_next

    J = dsys.J_np(x,[])
    Q = custom_process_noise(J, Q_c, dt, 1)

    x_priori, P_priori = kf.predict(dt, x, P, Q)

    for ob in ob_next_group:
        ob.update(x_priori, dsys.x_dict)
        ob.update_plot(x_priori, dsys.x_dict)

    nz = np.sum([ob.size for ob in ob_next_group])
    hx = create_observable_function(ob_next_group, dsys.x_dict, x_priori)

    y, R = get_all_measurements(ob_next_group, x_priori, dsys.x_dict)
    x, P = kf.update(y, R, hx)

    for ob in ob_next_group:
        ob.update_plot(x, dsys.x_dict)

    # Store results
    t_history.append(t_next)
    x_history.append(x.copy())
    P_history.append(P.copy())
    Q_history.append(Q)
    hx_history.append(hx)
    dt_history.append(dt)

    # Plot results
    body.plot()
    plot_track.update_state(x_history, t_history)
    #  plot_track.update_observation(y_history, t_history)
    print('Step no:',len(t_history),'complete')

    # Get next measurement time and observation group
    t_next, ob_next_group, nz = get_next_obs_group(obs_list,t,0.00001)

# ---------------------------------------------------------------------
# RTS smoothing on results
# ---------------------------------------------------------------------

x_rts, P_rts = kf.rts_smoother(x_history,
                               P_history,
                               Q_history,
                               dt_history,
                               quaternions=True)

plot_track.load_smoothed_data(x_rts, t_history)

input("(Press any key to close figures)")
plot_t.close()
plot_e.close()
plot_track.close()

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

x_history = np.array(x_history)
rts_history = np.array(x_rts)

raw_dict = {key: x_history[:,ind] for key, ind in dsys.x_dict.items()}
rts_dict = {key: rts_history[:,ind] for key, ind in dsys.x_dict.items()}

raw_df = pd.DataFrame(raw_dict)
rts_df = pd.DataFrame(rts_dict)

raw_df.to_csv('./results/'+NAME+"_raw.csv")
rts_df.to_csv('./results/'+NAME+"_rts.csv")

np.save('./results/P_kf.npy',np.array(P_history))
np.save('./results/P_rts.npy',np.array(P_rts))
