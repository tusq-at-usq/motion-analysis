#!/usr/bin/python3.8
"""Unscented Kalman Filter

Note that the variance of the data (R matrix) should match the variance of the error induced in the data.

Author: Andrew Lock
Date: 14/2/22
"""

import numpy as np
import pandas as pd
from scipy import linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from matplotlib import pyplot as plt


class UKF:
    def __init__(self,system,dt_int_max,true_data=None,self.dt=0.1):
        self.system = system
        self.dt_int_max = dt_int_max
        self.true_data = true_data
        self.x_history = []
        self.z_history = []
        self.var_history = []
        self.t=0
        self.t_history =[0]
        self.dt = dt

    def hx(self,X):
        Z = [X[i] for i in range(len(X)) if self.S.x[i] in self.S.observables]
        return Z

    def fx(self,X,dt,t):
        steps = int(np.ceil(dt/self.dt_int_max))
        dt_int = dt/steps 
        for di in steps:
            X = self.S.int_RK4(X,dti,t+di*dt_int)
        return X

    def x_mean_fn(self,sigmas,Wn):
        # Custom weighting function so we can normalise q
        x = np.zeros(len(sigmas[0]))
        for i in range(len(sigmas)):
            for j in range(len(sigmas[i])):
                    x[j] += sigmas[i,j]*Wn[i]
        x = self.normalise_q(x)
        return x

    def initialise(self,x0,P=None,Q=None,R=None,sigmas=None):
        # Instantiate the UKF class
        if sigmas is None:
            sigmas = MerweScaledSigmaPoints(self.S.nx, alpha=.001, beta=2., kappa=0.)

        f = UnscentedKalmanFilter(dim_x=self.S.nx, dim_z=self.S.nz,dt=self.dt,hx=self.hx, 
                                  fx=self.fx, points=sigmas,x_mean_fn=self.x_mean_fn)
        
        if P is None: 
            # Default ititial state covariance matrix
            P = np.eye(self.S.nx)*1 

        if R is None:
            # Default measurement covariance matrix
            R = np.eye((self.S.nz,self.S.nz))*0.1 # Data variance 

        if Q is None: 
            # Default process covariance matrix
            #TODO: A more precise method of Q should be used
            Q = np.eye((self.S.nx,self.S.nx))*0.01  

        f.x = np.array(self.S.x0) # Initial state
        f.P = P
        f.R = R
        f.Q = Q
        self.x_history.append(f.x)
        self.var_history.append(np.diag(f.P))
        self.z_history.append(self.hx(f.x))
        self.f = f

    def predict(self):
        self.f.predict()
        self.f.x = self.normalise_q(self.f.x)
        return self.hx(self.f.x)

    def update(self,z,dt=self.dt):
        self.f.x = self.normalise_q(self.f.x)
        self.f.update(z)
        self.f.x = self.normalise_q(self.f.x)
        self.x_history.append(self.f.x)
        self.var_history.append(np.diag(self.f.P))
        self.z_history.append(z)
        self.t += dt
        self.t_history.append(self.t)
        return self.f.x

    def normalise_q(self,x):
        q_inds = np.array([i for i,var in enumerate(self.S.x) if self.S.name_dict[var] in ["q0","q1","q2","q3"]]) 
        q_ = x[q_inds]
        if len(q_) not in  [0,4]: print("ERROR:",len(q),"quaternions identified)
        q_ = q_ / np.linalg.norm(q_)
        for qi,q_ind in zip(q_,q_inds):
            x[q_ind] = qi
        return x

