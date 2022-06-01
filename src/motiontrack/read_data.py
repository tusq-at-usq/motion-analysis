#!/usr/bin/python3.8
import os
import sys
import pandas as pd

class BlobData:
    # All blob data from file
    def __init__(self, data_file_name):
        self.data_file_name = data_file_name
        self.frame_start = 0
        self.frame_end   = 0
        self.region_list = []
        self.frames = {} # contains FrameData objects with frame numbers as keys
        self.variance = {}

        #TODO: Need to alter this to suit the input data format
        fileData = pd.read_csv(self.data_file_name,
                               header=0,
                               sep=" ",
                               index_col=False,
                               skipfooter=2) 

        # Find the names of headers for the important parameters (search for part of string)
        #TODO: Add more blob details if we have them
        frame_header = list(filter(lambda x: "Image" in x, fileData.keys()))[0]
        x_header = list(filter(lambda x: "xCord" in x, fileData.keys()))[0]
        y_header = list(filter(lambda x: "yCord" in x, fileData.keys()))[0]
        size_header = list(filter(lambda x: "size" in x, fileData.keys()))[0]

        self.frames = fileData[frame_header]
        self.X = fileData[x_header]
        self.Y = fileData[y_header]
        self.D = fileData[size_header]
        self.frame_start = np.min(fileData[frame_header])
        self.frame_end = np.max(fileData[frame_header])
        frames = np.unique(fileData[frame_header])
        for frame in frames:
            self.frames[frame] = FrameData(frame)
        for i,row in fileData.iterrows():
            self.frames[row[frame_header]].add_blob(row[x_header],row[y_header],row[size_header])

class FrameData:
    # Data for each frame
    def __init__(self, frame_id):
        self.n_particles = 0
        self.frame_id = frame_id
        self.X = []
        self.Y = []
        self.D = []

    def add_blob(self,x,y,D):
        self.n_particles += 1
        self.X.append(x)
        self.Y.append(-y)
        self.D.append(D)

    def get_blob_data(self):
        #TODO: Can incorporate more information if we have it 
        out_data = {
                'x'    : self.X,
                'y'    : self.Y,
                'size' : self.D
                }
        return out_data

    def plot_frame(self):
        fig,ax = plt.subplots()
        ax.plot(self.X,self.Y,'o')
        plt.show()

    def get_ind_blob_data(self, i):
        out_data = {
                'x'    : self.X[i],
                'y'    : self.Y[i],
                'size' : self.D[i]
                }
        return out_data
