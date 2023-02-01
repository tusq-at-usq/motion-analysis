
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
P_kf = np.load('P_kf.npy')
P_rts = np.load('P_rts.npy')
#  KF = 'drop_2_raw.csv'

TIMESTEP = 1/1000


fig, ax = plt.subplots(1, 1, sharex=True)
#  ax = ax.flatten()
for i in range(P_kf.shape[1]):
    ax.plot(P_kf[:,i,i],label=str(i))

ax.set_yscale('log')
ax.legend()
plt.tight_layout()

plt.show()
