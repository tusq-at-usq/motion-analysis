import numpy as np
from motiontrack.ukf import custom_process_noise


A = np.array(
    [[0,    1., 0,  0,  0,  0],
     [0,    0,  1,  0,  0,  0],
     [0,    0,  0,  0,  0,  0],
     [0,    0,  0,  0,  1,  0],
     [0,    0,  0,  0,  0,  1],
     [0,    0,  0,  0,  0,  0]]
)

Q_c = np.array(
    [[0,    0,  0,  0,  0,  0],
     [0,    0,  0,  0,  0,  0],
     [0,    0,  1,  0,  0,  0],
     [0,    0,  0,  0,  0,  0],
     [0,    0,  0,  0,  0,  0],
     [0,    0,  0,  0,  0,  1]])

Q_d = custom_process_noise(A, Q_c, 0.5, 1)
print(Q_d)

