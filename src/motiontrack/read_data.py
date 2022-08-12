#!/usr/bin/python3.8

""" Class and function to read blob data from file
"""

from typing import List
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class BlobsFrame:
    """ Class to represent 2D blob data found by image recognition
    TODO: better structure for this using dictionaries.
    """
    def __init__(self, points: np.array, diameters: np.array):
        self.points = points # x,y coordinates
        self.diameters = diameters # Diameters
        self.n = points.shape[1]

    def remove_blob(self, i: int):
        """
        Remove blob point from frame

        Parameters
        ----------
        i : int
            index of blob to remove
        """
        self.points = np.delete(self.points, i, axis=1)
        self.diameters = np.delete(self.diameters, i)
        self.n -= 1

    def plot_frame(self):
        """
        Plot the 2D blob frame

        """
        fig, ax = plt.subplots()
        ax.plot(self.points[0], self.points[1], 'o')
        plt.show()

class BlobFile:
    """ Stores all blob data from a file and processes it into frames
    """
    def __init__(self, filename: str):
        blob_data = pd.read_csv(filename,
                               header=0,
                               sep=" ",
                               index_col=False)

        # Find the names of headers for the important parameters (search for part of string)
        #TODO: Add more blob details if we have them
        self.frame_header = list(filter(lambda x: "Image" in x, blob_data.keys()))[0]
        self.x_header = list(filter(lambda x: "xCord" in x, blob_data.keys()))[0]
        self.y_header = list(filter(lambda x: "yCord" in x, blob_data.keys()))[0]
        self.size_header = list(filter(lambda x: "size" in x, blob_data.keys()))[0]

        self.name = filename.split("/")[-1].split(".")[0]
        self.data = blob_data
        self.frames = blob_data[self.frame_header]
        self.X = blob_data[self.x_header]
        self.Y = blob_data[self.y_header]
        self.D = blob_data[self.size_header]
        self.frame_start = np.min(blob_data[self.frame_header])
        self.frame_end = np.max(blob_data[self.frame_header])

def read_blob_data(filenames: List[str]) -> dict:
    """
    Reads one or more files containing 2D blob data, and produces a dictionary of
    BlobsFrames

    A BlobFrame instance is created for each video frame.
    All frames are stored within a BlobData object.

    Parameters
    ----------
    filenames : List[str]
        A list of filenames containing the blob data

    Returns
    -------
    dict
        A dictionary with structure:
        { <frame no.> : {<filename> : <BlobsFrame> } }

    """

    blob_files = []
    for filename in filenames:
        blob_files.append(BlobFile(filename))

    frame_start = np.min([f.frame_start for f in blob_files])
    frame_end = np.max([f.frame_end for f in blob_files])

    frame_dict = {frame:{} for frame in range(frame_start,frame_end)} # Dict of files

    for i in range(frame_start,frame_end):
        for file in blob_files:
            if i in file.data[file.frame_header]:
                data = file.data.loc[file.data[file.frame_header] == i]
                points = np.array([data[file.x_header],data[file.y_header]])
                diameters = np.array(data[file.size_header])
                frame_dict[i][file.name] = BlobsFrame(points, diameters)
    return frame_dict

