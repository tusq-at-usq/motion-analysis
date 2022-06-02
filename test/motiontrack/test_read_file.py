import numpy as np

from motiontrack.read_data import *

file = 'data/example_data.txt'
frames = read_files([file])
print(frames)
