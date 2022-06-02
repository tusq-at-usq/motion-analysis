#!/usr/bin/python3.8
import os
import sys
import pandas as pd
import numpy as np

class Blobs2D:
    # A set of 2D blob data from image data
    def __init__(self,p,D):
        self.p = p # x,y coordinates
        self.D = D # Diamters
        self.n = p.shape[1]
        # Add new data if used later

    def remove_blob(self,i):
        self.p = np.delete(self.p,i,axis=1)
        self.D = np.delete(self.D,i)
        self.n -= 1

    def plot_frame(self):
        fig,ax = plt.subplots()
        ax.plot(self.p[0],self.p[1],'o')
        plt.show()

class fileData:
    def __init__(self,filename):
        fileData = pd.read_csv(filename,
                               header=0,
                               sep=" ",
                               index_col=False) 

        # Find the names of headers for the important parameters (search for part of string)
        #TODO: Add more blob details if we have them
        self.frame_header = list(filter(lambda x: "Image" in x, fileData.keys()))[0]
        self.x_header = list(filter(lambda x: "xCord" in x, fileData.keys()))[0]
        self.y_header = list(filter(lambda x: "yCord" in x, fileData.keys()))[0]
        self.size_header = list(filter(lambda x: "size" in x, fileData.keys()))[0]

        self.name = filename.split("/")[-1].split(".")[0]
        self.data = fileData
        self.frames = fileData[self.frame_header]
        self.X = fileData[self.x_header]
        self.Y = fileData[self.y_header]
        self.D = fileData[self.size_header]
        self.frame_start = np.min(fileData[self.frame_header])
        self.frame_end = np.max(fileData[self.frame_header])


def read_files(filenames):
    # Iterates through frames from each 
    files = []
    for filename in filenames:
        files.append(fileData(filename))

    frame_start = np.min([f.frame_start for f in files])
    frame_end = np.max([f.frame_end for f in files])

    frameData = {frame:{} for frame in range(frame_start,frame_end)} # Dict of files

    for frame in range(frame_start,frame_end):
        for file in files:
            if frame in file.data[file.frame_header]:
                data = file.data.loc[file.data[file.frame_header] == frame]
                p = np.array([data[file.x_header],data[file.y_header]])
                D = np.array(data[file.size_header])
                frameData[frame][file.name] = Blobs2D(p,D)
    return frameData


