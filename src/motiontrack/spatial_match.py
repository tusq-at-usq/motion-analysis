#!/usr/bin/python3.8

"""Spatial match

Code to match location (x,y,z) and orientation (q0, q1, q2, q3) of an object
using blob location from data, and a model of the object.

Note this code is in 'draft' format. 

Both the code and algorithm could be significantly improved.

Author: Andrew Lock
Created: 30/5/22
"""
import time
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from getopt import getopt
from matplotlib import pyplot as plt

from motiontrack.utils import *

class SpatialMatch:
    def __init__(self,B,Vs):
        self.B = B
        self.Vs = Vs
        self.n_V = len(Vs)

    def run_match(self,
                  blob_data,
                  p_est,
                  Q_est,
                  plots = []):
        if len(blob_data) != self.n_V:
            print("ERROR: number of blob frames does not match viewpoints")
            raise ValueError
        p, Q = self.match_alg1(blob_data, p_est, Q_est, plots)
        plt.close()
        return p,Q

    def match_alg1(self,blob_data, p0, Q0, plots):
        # A simple traker which solves translation and rotation simultaneously
        # TODO: We NEED to develop better spatial match algorithms in the future

        # Extract the blob data from the  into 'unfiltered' lists
        X_ds_unfilt = []
        Y_ds_unfilt = []
        D_ds_unfilt = []
        for blobs in blob_data:
            X_d = blobs.points[:,0]
            Y_d = blobs.points[:,1]
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

            p_i = np.array([X_d, Y_d])
            plots[i].update_observation(p_i)

            # Update model and views
            p = np.array(p0)
            self.B.update(p,Q0)
            p_p = view.get_blobs().points.T

            D = np.array(D_d) # Diameters from image
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
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs)/np.linalg.norm(Qs)
            error = 0

            for i in range(self.n_V):
                view = self.Vs[i]
                X_d = X_ds[i]
                Y_d = Y_ds[i]
                D_d = D_ds[i]

                ps = p0 * C[4:7]
                self.B.update(ps,Qs)
                blobs = view.get_blobs()
                p_p = blobs.points

                blob_map = []
                min_norms = []
                for x,y,d in zip(X_d,Y_d,D_d): # Iterate through image blobs
                    # TODO: It wold be better to construct a matrix of distances
                    # and start from the closest blob. Instead of starting with 
                    # an arbitrary blob.
                    if len(p_p) > 0:
                        p_d = np.array([x,y])
                        delta_r = p_p - p_d
                        norms = np.linalg.norm(delta_r,axis=1) # Distance image blob and projected blobs
                        blob_map.append(np.argmin(norms))
                        j = np.argmin(norms)
                        p_p = np.delete(p_p.T,j,axis=1).T # Option: only allow each dot to be used once
                        min_norms.append(np.min(norms)) 
                        # OPTIONAL: Weight by surface norm
                rms = np.mean(np.array(min_norms)**2)**0.5
                min_norms = np.array(min_norms)[np.where(np.abs(min_norms)<2*rms)]
                min_norms = min_norms**2
                error_ = np.mean(min_norms[0:np.min((blobs.n,blobs.n))])
                error = error + error_
            return error

        def callback_plot(X):
            for view, X_d, Y_d, plot in zip(self.Vs, X_ds, Y_ds, plots):
                blobs_p = view.get_blobs()
                frame_p = view.get_mesh()
                cent = view.get_CoM()
                p_i = np.array([X_d, Y_d])
                plot.update_observation(p_i)
                plot.update_projection(blobs_p)
                plot.update_mesh(*frame_p)
                plot.update_CoM(cent)
                input("Press to continue")
                #  time.sleep(0.01)

        C0 = np.ones(7) 
        sol = minimize(get_cost,C0,method="Powell",callback=callback_plot, 
                       options={'xtol':1e-8,
                                'ftol':1e-6})

        C = sol.x
        Q_final = Q0 * C[0:4]
        Q_final = np.array(Q_final)/np.linalg.norm(Q_final)
        p_final = p0 * C[4:7]
        return p_final,Q_final
