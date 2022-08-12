""" Test module for reading blob data file and processing to a dictionary
of frames """

def test_read_file(print_flag=0):

    from motiontrack.read_data import read_blob_data

    FILENAMES = ['data/example_data.txt']
    frames = read_blob_data(FILENAMES)
    if print_flag:
        print(frames)

if __name__ == '__main__':
    test_read_file(1)
