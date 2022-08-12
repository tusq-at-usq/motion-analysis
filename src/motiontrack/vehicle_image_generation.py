#!/usr/bin/python3.8
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from getopt import getopt
import sys
import os

from motiontrack.utils import *
from motiontrack.read_data import BlobsFrame

""" View generation
DEPENDENCIES:
Python package "Pillow" (i.e. pip3 install Pillow)
Pyton package "ffmpeg" (i.e. pip3 install ffmpeg)
Linux package "ffmpeg" (i.e. sudo apt install ffmpeg)

The local level coordinate system is defined as:

    /|\ 1 (north)
     |
     |
     X--------> 2 (east)
  3 (into earth)

The vehicle/body coordinate system is defined as

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)
 
Authors: Andrew Lock
First created: Jan 2022

"""

def main(uoDict):
    filename = uoDict["--file"]
    if "--save" in uoDict:
        saveSwitch = True
    else:
        saveSwitch = False
    runFromCSV(filename,saveSwitch)

class View:
    # A projection of the 3D local coordinates to a specific 2D direction
    # TODO - can add a perspective projection option later if needed
    # TODO - a lot of this could be made more elegant without list loops
    def __init__(self,B,
                 viewAngle=[0,0,0],
                 name=None,
                 dirName="default",
                 saveSwitch=False,
                 scale=1,
                 perspective="parallel",
                 offset = 0):
        self.perspective=perspective
        self.viewAngle = viewAngle
        self.s_LV = [] # Vector from local origin to camera origin in local coordinates
        self.B = B
        self.surfPoints = [] # Updated at each timestep
        self.centPoint = [] # Body centroid point in camera view
        self.axisPoints = [] # Body axis points in camera view
        self.name = name
        self.dirName = dirName
        self.saveSwitch = saveSwitch
        self.artists = []
        self.blobSize = B.blobSize
        self.scale = scale  #Scale of the view image in pixels/m
        self.offset = offset # Placeholder offset
        self.initialisation()

    def initialisation(self):
        # Create a rotation matrix between local frame and camera frame
        q0,q1,q2,q3 = euler_to_quaternion(self.viewAngle[0],self.viewAngle[1],self.viewAngle[2])
        self.T_VL = quaternions_to_rotation_tensor(q0,q1,q2,q3)
        self.s_LV = np.matmul(np.array([0,0,-1]),self.T_VL)

    def initialise_plot(self):
        # A plot of the projection
        self.fig,self.ax = plt.subplots()
        self.ax.set_title("Projection:"+self.name)
        self.ax.set_facecolor('grey')
        self.ax.set_aspect('equal')

    def update(self):
        # 1. Update surface and blob projection coordinates from the BODY class 3D coordinates
        # 2. Determine which surfaces are visible
        # 3. Determine the blob size (dot product of surface normal and camera direction)
        #TODO: Can optimise this by using vectors instead of lists
        self.all_points = []
        self.all_blobs = []
        self.colours = []
        self.labels = []
        self.blob_sizes = []
        for surf in self.B.surfs:
            surfPoints = []
            surfBlobs = []
            if np.dot(self.s_LV,surf.n) > 0:
                points = np.array([p.XL for p in surf.points]+[surf.points[0].XL])
                for point in points:
                    surfPoints.append(np.matmul(self.T_VL,point))
                blobs = np.array([b.XL for b in surf.blobs])
                for blob in blobs:
                    surfBlobs.append(np.matmul(self.T_VL,blob))
                markersize = self.blobSize * np.dot(self.s_LV,surf.n)
                self.all_points.append(np.array(surfPoints))
                self.all_blobs.append(np.array(surfBlobs))
                self.blob_sizes.append(markersize)
                self.colours.append(surf.colour)
                self.labels.append(surf.name)
        self.centPoint = np.matmul(self.T_VL,self.B.X)
        self.axisPoints = [np.matmul(self.T_VL,ax.XL) for ax in self.B.axis]

    def get_2D_data(self):
        # Get 2D projection coordinates from 3D coordinates
        Xs = []
        Ys = []
        Ds = []
        for blobs,blobSize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                Xs.append(blobs[:,1])
                Ys.append(blobs[:,0])
                D = np.empty((blobs[:,1].size))
                D.fill(blobSize)
                Ds.append(D)
        Xs = np.array(Xs).flatten()
        Ys = np.array(Ys).flatten()
        Ds = np.array(Ds).flatten()
        return BlobsFrame(np.array([Xs,Ys]),Ds)

    def plot_frame(self):
        # Save a frame of an animation
        if not hasattr(self,"fig"):
            self.initialise_plot()
        # Create a frame of the animation
        ax = self.ax
        fillArtists = []
        for points,colour,label in zip(self.all_points,self.colours,self.labels):
            fill, = ax.fill(points[:,1],points[:,0],color=colour,label=label)
            fillArtists.append(fill)
        blobArtists = []
        for blobs,markersize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                blob, = ax.plot(blobs[:,1],blobs[:,0],'o',color='k',markersize=markersize)
                blobArtists.append(blob)
        self.artists.append(fillArtists + blobArtists)
        return fillArtists + blobArtists

    def plot_vehicle(self,title=None):
        # Plot a visualisation of the vehicle outline and blobs
        if not hasattr(self,"fig"):
            self.initialise_plot()
        self.ax.clear()
        self.ax.set_title(self.name)
        fillArtists = []
        for points,colour,label in zip(self.all_points,self.colours,self.labels):
            fill, = self.ax.fill(points[:,1],points[:,0],color=colour,label=label)
            points = np.vstack((points,points[0]))
            fill, = self.ax.plot(points[:,1],points[:,0],'-',color='k',label=label)
        for blobs,markersize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                blob, = self.ax.plot(blobs[:,1],blobs[:,0],'o',color='k',markersize=markersize*10)
        for axis,label in zip(self.axisPoints,['X','Y','Z']):
            #  line = self.ax.plot([axis[1],self.centPoint[1]],[axis[0],self.centPoint[0]])
            arrow = self.ax.arrow(self.centPoint[1],
                                  self.centPoint[0],
                                  axis[1]-self.centPoint[1],
                                  axis[0]-self.centPoint[0],
                                  width=0.01)
            self.ax.annotate(label, xy=(axis[1], axis[0]))

        if not title:
            title = self.name
        self.ax.set_title(title)
        plt.pause(0.01)
        self.fig.canvas.draw()

    def create_animation(self):
        # Compile the animation frames and save, if --save in options
        self.ax.set_facecolor('None')
        aniA = animation.ArtistAnimation(self.fig, self.artists)
        if self.saveSwitch:
            aniA.save(self.name+".avi",bitrate=200,dpi=100)
        return aniA

def runFromCSV(filename,saveSwitch):
    # Process data using a CSV file (only option for now)

    dirName = filename.split(".")[0]
    check_directory = os.path.isdir("./"+dirName)
    if not check_directory and saveSwitch:
        os.makedirs(dirName)

    resultData = pd.read_csv(filename,header=0,skiprows=[1,2]) # Read result data
    resultData = resultData.rename(columns=lambda x: x.strip()) # Remove white spaces from header names
    Xs = resultData["x_inertia"]
    Ys = resultData["y_inertia"]
    Zs = resultData["z_inertia"]
    q0s = resultData["q0"]
    q1s = resultData["q1"]
    q2s = resultData["q2"]
    q3s = resultData["q3"]
    ts = resultData["time"]

    scale = np.max(np.abs(np.array([Xs,Ys,Zs])))
    L = scale/5

    B = cubeGen(12)
    B.L = L

    # The default view angle is looking from above
    EA_above = np.array([0,0,0]) # Euler angle rotation for above
    EA_east = np.array([np.pi/2,np.pi/2,0]) # Euler angle rotation for local west view
    EA_north = np.array([np.pi,np.pi/2,0]) # Euler angle rotation for local north view

    viewA = View(B,EA_above,"above",dirName,saveSwitch)
    viewE = View(B,EA_east,"east",dirName,saveSwitch)
    viewN = View(B,EA_north,"north",dirName,saveSwitch)

    for q0,q1,q2,q3,X,Y,Z,t in zip(q0s,q1s,q2s,q3s,Xs,Ys,Zs,ts):

        # Update object location
        B.update([X,Y,Z],[q0,q1,q2,q3],t) # Update point locations

        # Update views
        viewA.update()
        viewE.update()
        viewN.update()

        # Plot animation frame
        viewA.plot_frame()
        viewE.plot_frame()
        viewN.plot_frame()
        # print(q0,q1,q2,q3)

    A = viewA.create_animation()
    E = viewE.create_animation()
    N = viewN.create_animation()
    plt.show()

def print_usage():
    print("Work in progress")
    print("===============")
    print("Currently defaults to unit cube shape")
    print("")
    print("  vehivle_image_generation.py --file=resultFile.csv")
    print("")
    print("Argument:                    Comment:")
    print("------------------------------------------------------------------------")
    print(" --file=               String containing result file name for results file.")
    print(" --save               Save the animation in a series of TIFF files.")
    print("")

def print_instructions():
    print("")
    print("     Tool to create images of trajectory")
    print("   ==============================================")
    print("Need to add instructions:")

short_options = ""
long_options = ["help", "file=", "save"]


if __name__ == '__main__':
    user_options = getopt(sys.argv[1:], short_options, long_options)
    uo_dict = dict(user_options[0])

    if "--instructions" in uo_dict:
        print_instructions()
        sys.exit(1)

    else:
        main(uo_dict)
        print("\n")
        print("SUCCESS.")
     
