""" Test module for reading blob data file and processing to a dictionary
of frames """

import os

from motiontrack.features import read_blob_data

FILENAME = os.path.dirname(__file__)+'/data/example_data.txt'

def test_read_file(print_flag=0):

    frames = read_blob_data([FILENAME])
    if print_flag:
        print(frames)

if __name__ == '__main__':
    test_read_file(1)
