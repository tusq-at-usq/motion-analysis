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

    def run_match(self,blob_data,p_est,Q_est,plot=0):
        if len(blob_data) != self.n_V:
            print("ERROR: number of blob frames does not match viewpoints")
            raise ValueError
        p,Q = self.match_alg1(blob_data,p_est,Q_est,plot)
        return p,Q

    def match_alg1(self,blob_data,p0,Q0,plot_flag=False):
        # A simple traker which solves translation and rotation simultaneously
        # TODO: We may develop alternative spatial match algorithms in the future

        # Extract the blob data from the  into 'unfiltered' lists
        X_ds_unfilt = []
        Y_ds_unfilt = []
        D_ds_unfilt = []
        for blobs in blob_data:
            X_d = blobs.points[0]
            Y_d = blobs.points[1]
            D_d = blobs.diameters
            X_ds_unfilt.append(X_d)
            Y_ds_unfilt.append(Y_d)
            D_ds_unfilt.append(D_d)

        # Filtering step - use a subselection of data blobs 
        # TODO: This step could probably be improved for robustness
        # TODO: Improve notation with 2D position vectors
        X_ds=[]; Y_ds=[]; D_ds=[]
        for i in range(self.n_V):
            view = self.Vs[i]
            X_d = X_ds_unfilt[i]
            Y_d = Y_ds_unfilt[i]
            D_d = D_ds_unfilt[i]

            # Update model and views
            p = np.array(p0) + view.offset 
            self.B.update(p,Q0)
            view.update()
            p_p = getattr(view.get_2D_data(),'points')

            D = np.array(D_d)*view.scale # Diameters from image
            p_p = p_p * view.scale # Blob vectors from projection
            p_d = np.array([X_d,Y_d]).T # Blob vectors from measured data

            # Find the distancer between each combination of blobs
            X_dif = np.subtract.outer(p_p[0],p_d.T[0])
            Y_dif = np.subtract.outer(p_p[1],p_d.T[1])
            r_dif = (X_dif**2 + Y_dif**2)**0.5

            # Find the closest image blob to each model blob
            # (Using each image blob only once)
            pairs = list(set([np.argmin(r) for r in r_dif]))

            # Only use the image blobs closest to the projected blobs 
            X_ds.append([X_d[i] for i in pairs])
            Y_ds.append([Y_d[i] for i in pairs])
            D_ds.append([D_d[i] for i in pairs])

        def get_cost(C):
            #TODO: Check this is the appropriate normalisation of quaternions. 
            # For a unit quaternion, should it just be the q1,q2,q3 which are normalised?
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs)/np.linalg.norm(Qs)
            error = 0
            #  for view,X_d,Y_d,D_d,offset in zip(self.views,X_ds,Y_ds,D_ds,self.offsets):
            for i in range(self.n_V):
                view = self.Vs[i]
                X_d = X_ds[i]
                Y_d = Y_ds[i]
                D_d = D_ds[i]

                ps = p0 * C[4:7]
                ps = ps + view.offset
                self.B.update(ps,Qs)
                view.update()
                blobs = view.get_2D_data()
                p_p = blobs.points

                blob_map = []
                min_norms = []
                p_p = p_p*view.scale
                for x,y,d in zip(X_d,Y_d,D_d): # Iterate through image blobs
                    # TODO: It wold be better to construct a matrix of distances
                    # and start from the closest blob. Instead of starting with 
                    # an arbitrary blob.
                    if len(p_p) > 0:
                        p_d = np.array([x,y])
                        delta_r = p_p.T - p_d
                        norms = np.linalg.norm(delta_r,axis=1) # Distance image blob and projected blobs
                        blob_map.append(np.argmin(norms))
                        j = np.argmin(norms)
                        p_p = np.delete(p_p,j,axis=1) # Option: only allow each dot to be used once
                        min_norms.append(np.min(norms)**2) 
                        # OPTIONAL: Weight by surface norm
                error_ = np.mean(min_norms[0:np.min((blobs.n,blobs.n))])
                error = error + error_
            return error

        if plot_flag:
            blob_fig,axs = plt.subplots(1,2)
            axs.flatten()
            def plot(X):
                for view,ax,X_d,Y_d,D_d in zip(self.Vs,axs,X_ds,Y_ds,D_ds):
                    ax.clear()
                    #  ax.scatter(X_d,Y_d,s=D,facecolors='none',edgecolor='k',)
                    blobs = view.get_2D_data()
                    p_p = blobs.points #* view.scale
                    D_p = blobs.diameters *500

                    proj = ax.scatter(p_p[0],p_p[1],s=D_p,color='r',label='projected')
                    ax.set_aspect("equal")
                    ax.set_title(view.name)

                    D_d = np.array(D_d)*view.scale*800
                    data = ax.scatter(X_d,Y_d,s=D_d,facecolors='none',edgecolor='k',label='data')
                plt.pause(0.01)
                blob_fig.canvas.draw()
                axs[1].legend()
                plt.tight_layout()

        C0 = np.ones(7) 
        if plot:
            sol = minimize(get_cost,C0,method="Powell",callback=plot)
            for view in self.Vs:
                view.plot_vehicle()
        else:
            sol = minimize(get_cost,C0,method="Powell")
        try:
            plt.pause(0.001)
            input("<Pres any key to close>")
            plt.close(blob_fig)
        except:
            pass

        C = sol.x
        Q_final = Q0 * C[0:4]
        Q_final = np.array(Q_final)/np.linalg.norm(Q_final)
        p_final = p0 * C[4:7]
        return p_final,Q_final
