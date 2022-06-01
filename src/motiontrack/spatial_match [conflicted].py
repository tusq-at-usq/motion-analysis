#!/usr/bin/python3.8

"""Spatial match

Code to match location (x,y,z) and orientation (q0, q1, q2, q3) of an object
using blob location from data, and a model of the object.

Author: Andrew Lock
Created: 30/5/22
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from getopt import getopt
from matplotlib import pyplot as plt

from motiontrack.utils import *
from motiontrack.read_data import *

class SpatialMatch:
    def __init__(self,B,Vs):
        self.B = B
        self.Vs = Vs
        self.n_V = len(Vs)

    def run_match(self,blob_data,r_est,Q_est,plot=0):
        if len(blob_data) != self.n_V:
            print("ERROR: number of blob frames does not match viewpoints")
            raise ValueError
        self.match_alg1(blob_data,r_est,Q_est,plot)
        # Code to find state X which matches blob locations.
        pass

    def match_alg1(self,blob_data,r0,Q0,plot_flag=False):
        # A simple traker which solves translation and rotation simultaneously
        # TODO: We may develop alternative spatial match algorithms in the future

        # Extract the blob data from the  into 'unfiltered' lists
        X_ims_unfilt = []
        Y_ims_unfilt = []
        D_ims_unfilt = []
        for blobs in blob_data:
            X_im = blobs.r[0]
            Y_im = blobs.r[1]
            D_im = blobs.D
            X_ims_unfilt.append(X_im)
            Y_ims_unfilt.append(Y_im)
            D_ims_unfilt.append(D_im)

        # Filtering step - use a subselection of data blobs 
        # TODO: This step could probably be improved for robustness
        X_ims=[]; Y_ims=[]; D_ims=[]
        for i in range(self.n_V):
            view = self.Vs[i]
            X_im = X_ims_unfilt[i]
            Y_im = Y_ims_unfilt[i]
            D_im = D_ims_unfilt[i]

            # Update model and views
            r = np.array(r0) + view.offset 
            self.B.update(r,Q0)
            view.update()
            r_i,D_i = getattr(view.get_2D_data(),'r','D')

            D = np.array(D_im)*view.scale # Diameters from image
            r_i = r_i * view.scale # Blob vectors from projection
            r_im = np.array([X_im,Y_im]).T # Blob vectors from measured data

            # Find the distancer between each combination of blobs
            X_dif = np.subtract.outer(r_i.T[0],r_im.T[0])
            Y_dif = np.subtract.outer(r_i.T[1],r_im.T[1])
            r_dif = (X_dif**2 + Y_dif**2)**0.5

            # Find the closest image blob to each model blob
            # (Using each image blob only once)
            pairs = list(set([np.argmin(r) for r in r_dif]))

            # Only use the image blobs closest to the projected blobs 
            X_ims.append([X_im[i] for i in pairs])
            Y_ims.append([Y_im[i] for i in pairs])
            D_ims.append([D_im[i] for i in pairs])

        def get_cost(C):
            #TODO: Check this is the appropriate normalisation of quaternions. 
            # For a unit quaternion, should it just be the q1,q2,q3 which are normalised?
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs)/np.linalg.norm(Qs)
            error = 0
            #  for view,X_im,Y_im,D_im,offset in zip(self.views,X_ims,Y_ims,D_ims,self.offsets):
            for i in range(self.n_V):
                view = self.Vs[i]
                X_im = X_ims[i]
                Y_im = Y_ims[i]
                D_im = D_ims[i]

                rs = r0 * C[4:7]
                rs = rs + view.offset
                self.B.update(rs,Qs)
                view.update()
                r_p,D_p = getattr(view.get_2D_data(),'r','D')

                blob_map = []
                min_norms = []
                r_ps = r_p*view.scale
                for x,y,d in zip(X_im,Y_im,D_im): # Iterate through image blobs
                    if len(r_ps) > 0:
                        r_im = [x,y]
                        breakpoint()
                        delta_r = r_ps - r_im
                        norms = np.linalg.norm(delta_r,axis=1) # Distance image blob and projected blobs
                        blob_map.append(np.argmin(norms))
                        j = np.argmin(norms)
                        # r_ps = np.delete(r_ps,j,axis=0) # Option: only allow each dot to be used once
                        min_norms.append(np.min(norms)**2) 
                        # TODO: Weight by surface norm
                error_ = np.mean(min_norms[0:np.min((len(X_im),len(X_p)))])
                error = error + error_
            return error

        if plot_flag:
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
        r_final = r0 * C[4:7]
        return r_final,Q_final
