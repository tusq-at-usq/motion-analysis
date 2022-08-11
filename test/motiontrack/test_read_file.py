""" Test module for reading blob data file and processing to a dictionary
of frames """

from motiontrack.read_data import read_blob_data

FILENAMES = ['data/example_data.txt']
frames = read_blob_data(FILENAMES)
print(frames)
