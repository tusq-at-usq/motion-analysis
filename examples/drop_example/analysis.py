
""" Quick analysis of tracking results
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from motiontrack.utils import quaternion_to_euler

#  RAW = "run1327_raw.csv"
#  RTS = "run1327_rts.csv"

#  RAW = "run1329_raw.csv"
#  RTS = "run1329_rts.csv"
#  RTS = "run1329_aero_rts.csv"
#  RTS = "drop_1.csv"
#  RTS = "run1329_rts.csv"

#  RTS = 'drop_2_raw.csv'
RTS = './results/drop_2_rts.csv'
KF = './results/drop_2_raw.csv'
#  KF = 'drop_2_raw.csv'

TIMESTEP = 1/1000

rts_df = pd.read_csv(RTS)
pos = rts_df[['x', 'y', 'z']].to_numpy()
acc_rts = rts_df[['a_x', 'a_y', 'a_z']].to_numpy()
vel = np.diff(pos, axis=0)/TIMESTEP
acc = np.diff(vel, axis=0)/TIMESTEP
Q = rts_df[['q0', 'q1', 'q2', 'q3']].to_numpy().T
euler = np.array(quaternion_to_euler(*Q)).T
omega = np.diff(euler, axis=0)/TIMESTEP

kf_df = pd.read_csv(KF)
pos_kf = kf_df[['x', 'y', 'z']].to_numpy()
acc_kf = kf_df[['a_x', 'a_y', 'a_z']].to_numpy()
vel_kf_dif = np.diff(pos_kf, axis=0)/TIMESTEP
acc_kf_dif = np.diff(vel_kf_dif, axis=0)/TIMESTEP
Q_kf = kf_df[['q0', 'q1', 'q2', 'q3']].to_numpy().T
euler_kf = np.array(quaternion_to_euler(*Q_kf)).T
omega_kf = np.diff(euler_kf, axis=0)/TIMESTEP


n = len(pos[:,0])

t_r = np.linspace(0,TIMESTEP*n, n)
t_v = np.linspace(TIMESTEP/2,TIMESTEP*n - TIMESTEP/2, n-1)
t_a = np.linspace(TIMESTEP,n*TIMESTEP - TIMESTEP, n-2)

fig, ax = plt.subplots(2, 1, sharex=True)
ax = ax.flatten()
ax[0].plot(t_r, pos[:,0], '.-', linewidth=1, label=r'$s_x$')
ax[0].plot(t_r, pos[:,1], '.-', linewidth=1, label=r'$s_y$')
ax[0].plot(t_r, pos[:,2], '.-', linewidth=1, label=r'$s_z$')

ax[1].plot(t_v, omega[:,0], label=r'$p$')
ax[1].plot(t_v, omega[:,1], label=r'$q$')
ax[1].plot(t_v, omega[:,2], label=r'$r$')
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel("Time [s]")
ax[0].set_ylabel("Position [m]")
ax[1].set_ylabel("Orientation [rad]")

fig, ax = plt.subplots(2, 1, sharex=True)
ax = ax.flatten()
ax[0].plot(t_v, vel[:,0], '.-', linewidth=1, label=r'$v_x$')
ax[0].plot(t_v, vel[:,1], '.-', linewidth=1, label=r'$v_y$')
ax[0].plot(t_v, vel[:,2], '.-', linewidth=1, label=r'$v_z$')

ax[1].plot(t_r, euler[:,0], label=r'$\psi$')
ax[1].plot(t_r, euler[:,1], label=r'$\theta$')
ax[1].plot(t_r, euler[:,2], label=r'$\phi$')
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel("Time [s]")
ax[0].set_ylabel("Velocity [m/s]")
ax[1].set_ylabel("Orientation [rad]")


fig, ax = plt.subplots(2, 1, sharex=True)
ax = ax.flatten()
ax[0].plot(t_a, acc[:,0], ':', linewidth=1, label=r'$a_x$')
ax[0].plot(t_a, acc[:,1], ':', linewidth=1, label=r'$a_y$')
ax[0].plot(t_a, acc[:,2], ':', linewidth=1, label=r'$a_z$')

ax[0].plot(t_r, acc_rts[:,0], '--', linewidth=1)
ax[0].plot(t_r, acc_rts[:,1], '--', linewidth=1)
ax[0].plot(t_r, acc_rts[:,2], '--', linewidth=1)

#  ax[0].plot(t_r, acc_kf[:,0], '-.', linewidth=1)
#  ax[0].plot(t_r, acc_kf[:,1], '-.', linewidth=1)
ax[0].plot(t_r, acc_kf[:,2], '-.', linewidth=1)

#  ax[0].plot(t_a, acc_kf_dif[:,0], ':', linewidth=1)
#  ax[0].plot(t_a, acc_kf_dif[:,1], ':', linewidth=1)
#  ax[0].plot(t_a, acc_kf_dif[:,2], ':', linewidth=1)

ax[1].plot(t_r, euler[:,0], label=r'$\psi$')
ax[1].plot(t_r, euler[:,1], label=r'$\theta$')
ax[1].plot(t_r, euler[:,2], label=r'$\phi$')
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel(r"Time [s]")
ax[0].set_ylabel(r"Acceleration [m/s$^2$]")
ax[1].set_ylabel(r"Orientation [rad]")
plt.tight_layout()

#  jerk_rts = rts_df[['j_x', 'j_y', 'j_z']].to_numpy()
#  fig, ax = plt.subplots(1, 1, sharex=True)
#  ax.plot(t_r, jerk_rts[:,0], '--', linewidth=1)
#  ax.plot(t_r, jerk_rts[:,1], '--', linewidth=1)
#  ax.plot(t_r, jerk_rts[:,2], '--', linewidth=1)




#  fig1, ax1 = plt.subplots(1,1)
#  ax1.plot(np.abs(cos(V
plt.show()
