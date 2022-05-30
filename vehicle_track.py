#!/usr/bin/python3.8
"""
Code to track particles using output data from imageAnalyse.py

A modified version of ParticleTracker.py by Viv Bone, specifically for use with vehicle tracking

Author: Andrew Lock 
Date: 14/2/22

"""

import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statistics as st
import pickle
import csv
import pandas as pd
from scipy.optimize import minimize

from getopt import getopt

from idmoc.vehicle_master.vehicle_image_generation import *

class BlobData:
    # All blob data from file
    def __init__(self, data_file_name):
        self.data_file_name = data_file_name
        self.frame_start = 999999999
        self.frame_end   = 0
        self.region_list = []
        self._data = {} # contains FrameData objects with frame numbers as keys
        self.variance = {}

        if "disturbed" in data_file_name:
            fileData = pd.read_csv(self.data_file_name, header=0, sep=" ",index_col=False, skipfooter=2) # Read result data
        else:
            fileData = pd.read_csv(self.data_file_name, header=5, skiprows=[6, 7,-1,-2], sep=" ",index_col=False, skipfooter=2) # Read result data

        # Find the names of headers for the important parameters
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
            self._data[frame] = FrameData(frame)
        for i,row in fileData.iterrows():
            self._data[row[frame_header]].add_blob(row[x_header],row[y_header],row[size_header])

class FrameData:
    # Data for each frame
    def __init__(self, frame_id):
        self.n_particles = 0
        self.frame_id = frame_id
        self._data = np.zeros((0,4))
        self.X = []
        self.Y = []
        self.D = []

    def add_blob(self,x,y,D):
        self.n_particles += 1
        self.X.append(x)
        self.Y.append(-y)
        self.D.append(D)

    def get_blob_data(self):
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

class Tracker6DoF:
    def __init__(self):
        self.offsets = np.zeros(7)
        self.true_data = None
        self.calculate_offsets = True # Default, can be overwritten
        self.x0 = np.zeros(13)
        self.data_files = []
        self.views = []
        self.B = None
        self.t = 0
        self.result_filename = "tracker_results.csv"
        self.dt_image = None # Integration timestep
        self.dt_int_max = None

    def initialize_tracker(self):
        blob_data_list = []
        for filename in self.data_files:
            blob_data_list.append(BlobData(filename))
        self.blob_data_list = blob_data_list
        #TODO: This structure is inherited and could be improved
        frame_sets = [[blob_data._data[i] for i in 
                        range(blob_data.frame_start, blob_data.frame_end)] 
                        for blob_data in self.blob_data_list]
        self.frame_sets = frame_sets
        if self.calculate_offsets:
            # Calculate the translation offset between the two images 
            # (starting XYZ between two views is not necessarily aligned)
            print("CALCULATING IMAGE OFFSETS")
            Q0 = self.x0[6:10]
            XYZ0 = self.x0[10:13]
            self.determine_offsets(Q0,XYZ0)
            # TODO:  Could be improved -there are redundant numbers in each 
            # because the viewpoints are normal to the starting surface 
            # (information loss)
        self.dt_int = int(np.ceil(self.dt_image/self.dt_int_max))

    def euler_to_quaternion(self,psi, theta, phi):
        """Convert Euler angles to quaternions."""
        # According to Zipfel Eqn. 10.12
        q0 = np.cos(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        q1 = np.cos(psi/2) * np.cos(theta/2) * np.sin(phi/2) \
            - np.sin(psi/2) * np.sin(theta/2) * np.cos(phi/2)
        q2 = np.cos(psi/2) * np.sin(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.cos(theta/2) * np.sin(phi/2)
        q3 = np.sin(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            - np.cos(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        return q0, q1, q2, q3

    def determine_offsets(self,Q0,XYZ0):
        frames = [self.frame_sets[0][2],self.frame_sets[1][2]]
        Q1,XY1 = self.offset_tracker(self.views[0],frames[0],Q0,XYZ0)
        Q2,XY2 = self.offset_tracker(self.views[1],frames[1],Q0,list(XY1))

        # self.offsets = np.array([XY1,XY2-XY1])
        self.offsets = np.array([XY1,XY2])
        print(self.offsets)

    def offset_tracker(self,view,frame,Q0,XYZ0):
        # A tracker which has nested blob-tracking, and separates translation and rotation
        X_ims = []
        Y_ims = []
        image_data = frame.get_blob_data()
        X_im = image_data["x"]
        Y_im = image_data["y"]
        D_im = image_data["size"]
        vec_im = np.array(X_im + Y_im + D_im)
        scale0 = view.scale

        def get_XYZscale(X_):
            Xs = XYZ0 + X_[0:3]
            scale = scale0 * X_[3]
            error = 0
            # for view,X_im,Y_im,offset in zip(self.views,X_ims,Y_ims,self.offsets):
            self.B.update(Xs,Q0,0)
            view.update()
            X_p,Y_p,D_p = view.get_2D_data()
            # Find the closest projection point to each of the points in the image recognition
            blob_map = [] # Closest projection blob to each image blob
            min_norms = []
            r_ps = np.array([X_p,Y_p]).T
            r_ps = r_ps*scale
            for x,y,d in zip(X_im,Y_im,D_im):
                if len(r_ps) > 0:
                    r_im = [x,y]
                    delta_r = r_ps - r_im
                    norms = np.linalg.norm(delta_r,axis=1)
                    blob_map.append(np.argmin(norms))
                    r_ps = np.delete(r_ps,np.argmin(norms),axis=0) # Only allow each dot to be used once
                    min_norms.append(np.min(norms)**2)
            min_norms = np.sort(np.array(min_norms))
            error = np.sum(min_norms[0:np.min((len(X_im),len(X_p)))])
            return error

        X0 = [1,1,1,1]
        # sol = minimize(get_Q_cost,X_outer0,args=(XYZ0,0),method="Nelder-Mead",options={"xatol":0.002,"fatol":10})
        sol = minimize(get_XYZscale,X0,method="Powell")
        # sol = minimize(get_Q_cost,X_outer0,args=(XYZ0,0))
        view.scale = sol.x[3] * view.scale
        print("View",view.name,"scale=",view.scale)
        Q_final = Q0 # Redundant
        XYZ_final = sol.x[0:3] + XYZ0

        # self.B.update(XYZ_final,Q0,0)
        # view.update()
        # fig,axs = plt.subplots(1,1)
        # axs.plot(X_im,Y_im,"o",color='k')
        # X_p,Y_p,D_p = view.get_2D_data()
        # r_ps = np.array([X_p,Y_p]).T
        # r_ps = r_ps * view.scale
        # axs.plot(r_ps[:,0],r_ps[:,1],"o",color='r')
        # axs.set_aspect("equal")
        # axs.set_title(view.name)
            # # view.plot_vehicle()
        # plt.pause(0.0001)
        # plt.draw()
        # input("Alignment check for view"+view.name)
        # plt.close(fig)
        return Q_final,XYZ_final

    def simple_tracker(self,i,Q0,XYZ0,plot=False):
        # A simple traker which solves translation and rotation simultaneously
        frames = [frame_set[i] for frame_set in self.frame_sets]
        X_ims = []
        Y_ims = []
        D_ims = []
        for frame in frames:
            image_data = frame.get_blob_data()
            X_im = image_data["x"]
            Y_im = image_data["y"]
            D_im = image_data["size"]
            vec_im = np.array(X_im + Y_im + D_im)
            n = vec_im.size

            # Filter for outliers
            r_ims = np.array([X_im,Y_im]).T
            im_norms = np.linalg.norm(r_ims,axis=1)
            norm_std = np.std(im_norms)
            norm_mean = np.mean(im_norms)
            X_im = [X_im[i] for i in range(len(X_im)) if np.abs(im_norms[i]-norm_mean) <  3*norm_std]
            Y_im = [Y_im[i] for i in range(len(Y_im)) if np.abs(im_norms[i]-norm_mean) <  3*norm_std]
            D_im = [D_im[i] for i in range(len(D_im)) if np.abs(im_norms[i]-norm_mean) <  3*norm_std]
            X_ims.append(X_im)
            Y_ims.append(Y_im)
            D_ims.append(D_im)

    # Code here to estiamte position region and filter blobs

        def get_cost(C):
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs)/np.linalg.norm(Qs)
            XYs = XYZ0 * C[4:7]
            error = 0
            for view,X_im,Y_im,D_im,offset in zip(self.views,X_ims,Y_ims,D_ims,self.offsets):
                XYs = XYs + offset
                self.B.update(XYs,Qs,self.t)
                view.update()
                X_p,Y_p,D_p = view.get_2D_data()
                # Find the closest projection point to each of the points in the image recognition
                blob_map = []
                min_norms = []
                r_ps = np.array([X_p,Y_p]).T
                r_ps = r_ps*view.scale
                for x,y,d in zip(X_im,Y_im,D_im):
                    if len(r_ps) > 0:
                        r_im = [x,y]
                        delta_r = r_ps - r_im
                        norms = np.linalg.norm(delta_r,axis=1)
                        try:
                            blob_map.append(np.argmin(norms))
                            j = np.argmin(norms)
                        except:
                            breakpoint()
                        r_ps = np.delete(r_ps,j,axis=0) # Only allow each dot to be used once
                        min_norms.append(np.min(norms)**2)
                error_ = np.sum(min_norms[0:np.min((len(X_im),len(X_p)))])
                error = error + error_
            return error

        if plot:
            blob_fig,axs = plt.subplots(1,2)
            axs.flatten()
            def plot(X):
                for view,ax,X_im,Y_im,D_im in zip(self.views,axs,X_ims,Y_ims,D_ims):
                    ax.clear()
                    D = np.array(D_im)*view.scale
                    ax.scatter(X_im,Y_im,s=D,facecolors='none',edgecolor='k',)
                    X_p,Y_p,D_p = view.get_2D_data()
                    r_ps = np.array([X_p,Y_p]).T
                    r_ps = r_ps * view.scale
                    ax.scatter(r_ps[:,0],r_ps[:,1],color='r')
                    ax.set_aspect("equal")
                    ax.set_title(view.name)
                plt.pause(0.001)
                blob_fig.canvas.draw()

        C0 = np.ones(7) 
        if plot:
            sol = minimize(get_cost,C0,callback=plot,tol=0.001,options={"eps":0.01})
        else:
            # sol = minimize(get_cost,C0,tol=0.00001,options={"eps":0.01})
            sol = minimize(get_cost,C0,method="Powell")
            # sol = minimize(get_cost,C0)
        # for view in self.views:
            # # view.update()
            # view.plot_vehicle()
        try:
            plt.pause(0.001)
            plt.close(blob_fig)
        except:
            pass

        C = sol.x
        Q_final = Q0 * C[0:4]
        Q_final = np.array(Q_final)/np.linalg.norm(Q_final)
        XY_final = XYZ0 * C[4:7]
        return Q_final, XY_final

    def simple_tracker2(self,i,Q0,XYZ0,plot=False):
        # A simple traker which solves translation and rotation simultaneously
        frames = [frame_set[i] for frame_set in self.frame_sets]
        X_ims_unfilt = []
        Y_ims_unfilt = []
        D_ims_unfilt = []
        for frame in frames:
            image_data = frame.get_blob_data()
            X_im = image_data["x"]
            Y_im = image_data["y"]
            D_im = image_data["size"]
            X_ims_unfilt.append(X_im)
            Y_ims_unfilt.append(Y_im)
            D_ims_unfilt.append(D_im)

    # Code here to estiamte position region and filter blobs
        radius = 10
        # X_is=[]; Y_is=[]; D_is=[]
        # for view,offset in zip(self.views,self.offsets):
            # XYZ = np.array(XYZ0) + offset
            # self.B.update(XYZ,Q0,self.t)
            # view.update()
            # X_i,Y_i,D_i = view.get_2D_data()
            # X_is.append(X_i)
            # Y_is.append(Y_i)
            # D_is.append(D_i)
        # blob_fig,axs = plt.subplots(1,2)
        X_ims=[]; Y_ims=[]; D_ims=[]
        # axs.flatten()
        # for view,offset,X_im,Y_im,D_im,ax in zip(self.views,self.offsets,X_ims_unfilt,Y_ims_unfilt,D_ims_unfilt,axs):
        for view,offset,X_im,Y_im,D_im in zip(self.views,self.offsets,X_ims_unfilt,Y_ims_unfilt,D_ims_unfilt):
            XYZ = np.array(XYZ0) + offset
            self.B.update(XYZ,Q0,self.t)
            view.update()
            X_i,Y_i,D_i = view.get_2D_data()
            # ax.clear()
            D = np.array(D_im)*view.scale
            # ax.scatter(X_im,Y_im,s=D,facecolors='none',edgecolor='k',)
            r_i = np.array([X_i,Y_i]).T
            r_im = np.array([X_im,Y_im]).T
            r_i = r_i * view.scale
            # ax.scatter(r_i[:,0],r_i[:,1],facecolors='None',edgecolor='k',s=20)
            # ax.set_aspect("equal")
            # Find the closest match for each pair
            X_dif = np.subtract.outer(r_i.T[0],r_im.T[0])
            Y_dif = np.subtract.outer(r_i.T[1],r_im.T[1])
            r_dif = (X_dif**2 + Y_dif**2)**0.5

            # r_mins1 = [np.min(r) for r in r_dif.T if np.min(r) < radius]
            # pairs = [i for i,r in enumerate(r_dif.T) if np.min(r) < radius]
            
            r_mins = [np.min(r) for r in r_dif]
            pairs = [np.argmin(r) for r in r_dif]
            # breakpoint()
            
            # print("------------------------")
            # print("Pairs:",list(set(pairs)))
            # print("Pairs1:",list(set(pairs1)))
            
            #TODO: Change to points within the radius, not the closest points
            # for r,pair in zip(r_i,pairs):
                # ax.plot(X_im[pair],Y_im[pair],'x')
                # ax.plot([r[0],X_im[pair]],[r[1],Y_im[pair]],color='b')
            # for r,pair in zip(r_i,pairs1):
                # ax.plot(X_im[pair],Y_im[pair],'^')
                # # ax.plot([r[0],X_im[pair]],[r[1],Y_im[pair]],color='r')
            pairs = list(set(pairs))
            X_ims.append([X_im[i] for i in pairs])
            Y_ims.append([Y_im[i] for i in pairs])
            D_ims.append([D_im[i] for i in pairs])
            # breakpoint()
        # plt.pause(0.001)
        # plt.draw()

        def get_cost(C):
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs)/np.linalg.norm(Qs)
            error = 0
            for view,X_im,Y_im,D_im,offset in zip(self.views,X_ims,Y_ims,D_ims,self.offsets):
                XYs = XYZ0 * C[4:7]
                XYs = XYs + offset
                self.B.update(XYs,Qs,self.t)
                view.update()
                X_p,Y_p,D_p = view.get_2D_data()
                # Find the closest projection point to each of the points in the image recognition
                blob_map = []
                min_norms = []
                r_ps = np.array([X_p,Y_p]).T
                r_ps = r_ps*view.scale
                for x,y,d in zip(X_im,Y_im,D_im):
                    if len(r_ps) > 0:
                        r_im = [x,y]
                        delta_r = r_ps - r_im
                        norms = np.linalg.norm(delta_r,axis=1)
                        try:
                            blob_map.append(np.argmin(norms))
                            j = np.argmin(norms)
                        except:
                            breakpoint()
                        # r_ps = np.delete(r_ps,j,axis=0) # Only allow each dot to be used once
                        min_norms.append(np.min(norms)**2)
                error_ = np.sum(min_norms[0:np.min((len(X_im),len(X_p)))])
                error = error + error_
            return error

        if plot:
            blob_fig,axs = plt.subplots(1,2)
            axs.flatten()
            def plot(X):
                for view,ax,X_im,Y_im,D_im in zip(self.views,axs,X_ims,Y_ims,D_ims):
                    ax.clear()
                    D = np.array(D_im)*view.scale
                    ax.scatter(X_im,Y_im,s=D,facecolors='none',edgecolor='k',)
                    X_p,Y_p,D_p = view.get_2D_data()
                    r_ps = np.array([X_p,Y_p]).T
                    r_ps = r_ps * view.scale
                    ax.scatter(r_ps[:,0],r_ps[:,1],color='r')
                    ax.set_aspect("equal")
                    ax.set_title(view.name)
                plt.pause(0.001)
                blob_fig.canvas.draw()

        C0 = np.ones(7) 
        if plot:
            # sol = minimize(get_cost,C0,callback=plot,tol=0.001,options={"eps":0.01})
            sol = minimize(get_cost,C0,method="Powell",callback=plot)
            for view in self.views:
                # view.update()
                view.plot_vehicle()
        else:
            # sol = minimize(get_cost,C0,tol=0.00001,options={"eps":0.01})
            sol = minimize(get_cost,C0,method="Powell")
            # sol = minimize(get_cost,C0)
        try:
            plt.pause(0.001)
            plt.close(blob_fig)
        except:
            pass

        C = sol.x
        Q_final = Q0 * C[0:4]
        Q_final = np.array(Q_final)/np.linalg.norm(Q_final)
        XY_final = XYZ0 * C[4:7]
        return Q_final, XY_final

    def quaternion_multiply(self,quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def write_data_file_header(self):
        string0 = "frame, time"  # 1st row is labels
        string1 = " na , [s] "
        string2 = "na, na"
        for i, name in enumerate(self.vehicle.X_map):
            string0 +=", {0}".format(name)
            if self.vehicle.X_details is None:
                string1 += ", na"
                string2 += ", na"
            else:
                string1 += ", {0}".format(self.vehicle.X_details[name][0])
                string2 += ", {0}".format(self.vehicle.X_details[name][1])
        string0 += "\n"
        string1 += "\n"
        string2 += "\n"
        with open(self.result_filename, 'w') as f:
            f.write(string0)
            f.write(string1)
            f.write(string2)

    def write_data_line(self,i):
        string = '{:.1e}'.format(i)
        string += ', {:.12e}'.format(self.t)
        for state in self.filter.x_history[-1]:
            string += ', {:.12e}'.format(state)
        for raw_data in self.filter.z_history[-1]:
            string += ', {:.12e}'.format(raw_data)
        for covariance in self.filter.var_history[-1]:
            string += ', {:.12e}'.format(covariance)
        string += "\n"
        with open(self.result_filename, 'a') as f:
            f.write(string)

def z_noise(z,scale):
    zq = [q+np.random.normal(scale=scale/4) for q in z[0:4]]
    zq = list(np.array(zq)/np.linalg.norm(zq))
    zX = [x*(1+np.random.normal(scale=scale)) for x in z[4:7]]
    return zq+zX


def main(uo_dict):
    T = Tracker6DoF()
    exec(open(uo_dict["--job"]).read(), globals(),locals())

    print("INITIALISING")
    T.initialize_tracker()
    T.write_data_file_header()

    print("STARTING VECHICLE TRACKING")
    for i in range(3,len(T.frame_sets[0])):
        try:
            # Q0, XYZ0 = nested_tracker(B,views,frames,0,Q0,XYZ0,offsets)
            print("Frame No:",i)
            prior = T.filter.predict()
            Q0 = prior[0:4]
            XYZ0 = prior[4:7]
            # Q, XYZ = T.simple_tracker(i,Q0,XYZ0,plot=True)
            # Q, XYZ = T.simple_tracker(i,Q0,XYZ0)
            if "--plot-blob" in uo_dict:
                Q, XYZ = T.simple_tracker2(i,Q0,XYZ0,plot=True)
            else:
                Q, XYZ = T.simple_tracker2(i,Q0,XYZ0)
            z = list(Q)+list(XYZ)
            if "--sim-error" in uo_dict:
                print("Simulating data error")
                z = z_noise(z,0.05)

            T.filter.update(z)
            T.filter.plot()
            T.write_data_line(i)
        except Exception as e:
            print(e)
            plt.show()
            breakpoint()
    plt.show()

#--------------------------------------------------------------------------

def print_usage():
    print("Work in progress")
    print("===============")
    print("Code for tracking vehicle movement using iamge data")
    print("")
    print(" vehicle_track.py --job=jobfile    ")
    print("")
    print("Argument:                    Comment:")
    print("------------------------------------------------------------------------")
    print(" --job=               String containing file name for the job file.")
    print(" --sim-error          Simulate Guassian error in data.")
    print(" --plot-blob          Plot blob match at each step.")
    print("")

short_options = ""
long_options = ["help", "job=","out-file=","sim-error","plot-blob"]

if __name__ == "__main__":
    user_options = getopt(sys.argv[1:], short_options, long_options)
    uo_dict = dict(user_options[0])

    if len(user_options[0]) == 0 or "--help" in uo_dict:
        print_usage()
        sys.exit(1)

    else:
        #  try:
        main(uo_dict)
        print("\n")
        print("SUCCESS.")
        print("\n")

        #  except Exception as e:
            #  print("\nERROR: " + e)
            #  sys.exit(1)

