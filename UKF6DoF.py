#!/usr/bin/python3.8
"""6DoF Unscented Kalman Filter

Can be used with a vehicle class to filter results.
The process model is drawn from vehicle.get_X with an RK4 integrator

Requires the FilterPy package, which can be installed via pip (pip3 install filterpy)

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


class UKF6DoF:
    def __init__(self,vehicle,dt_data,dt_int_max,true_data=None):
        self.vehicle = vehicle
        self.dt_int_max = dt_int_max
        self.true_data = true_data
        self.z_names = ["q0","q1","q2","q3","x_inertia","y_inertia","z_inertia"]
        self.x_history = []
        self.z_history = []
        self.var_history = []
        self.t=0
        self.t_history =[0]
        self.fig, self.ax = plt.subplots(2,2,figsize=(10,8))
        self.ax = self.ax.flatten()
        self.dt_data = dt_data
        self.dt_multiple  = None
        self.dt_int = None

    def hx(self,X):
        Z = []
        for var_name in self.z_names:
            Z.append(X[self.vehicle.X_map[var_name]])
        Z = np.array(Z)
        return Z

    def fx(self,X,dt_data):
        dt = dt_data/self.dt_multiple
        for _ in range(self.dt_multiple):
            X = RK4(self.vehicle,X,dt,0)
        return X

    def x_mean_fn(self,sigmas,Wn):
        x = np.zeros(len(sigmas[0]))
        for i in range(len(sigmas)): # Each sigma point
            for j in range(len(sigmas[i])): # Each state
                    x[j] += sigmas[i,j]*Wn[i]
        x = self.normalise_q(x)
        return x

    def initialise(self,x0,P=None,Q=None,R=None,sigmas=None):
        # Instantiate the UKF class
        # Default is a system with 13 states. 
        # Assumes that the initialisation is first Z, with guesses for remaining X
        if sigmas is None:
            sigmas = MerweScaledSigmaPoints(13, alpha=.001, beta=2., kappa=0.)

        f = UnscentedKalmanFilter(dim_x=13, dim_z=7,dt=self.dt_data,hx=self.hx, 
                                  fx=self.fx, points=sigmas,x_mean_fn=self.x_mean_fn)
        
        if P is None: 
            P = np.eye(13)*1000 # Initial covariance matrix
            P[6:10,6:10] = np.eye(4)*0.4
            P[3:6,3:6] = np.eye(3)*10

        if R is None:
            R = np.zeros((7,7)) # Data variance 
            R[0:4,0:4] = np.eye(4)*0.05
            R[4:7,4:7] = np.eye(3)*10

        if Q is None: 
            Q = np.ones((13,13))*0.000001 # Process variance 
            #TODO: A more precise method of Q should be used

        f.x = np.array(x0) # Initial state
        f.P = P
        f.R = R
        f.Q = Q
        self.x_history.append(f.x)
        self.var_history.append(np.diag(f.P))
        self.z_history.append([x0[self.vehicle.X_map[zi]] for zi in self.z_names])
        self.f = f
        self.dt_multiple = int(np.ceil(self.dt_data/self.dt_int_max))
        self.dt_int = self.dt_multiple * self.dt_data

    def predict(self):
        self.f.predict()
        self.f.x = self.normalise_q(self.f.x)
        return self.hx(self.f.x)

    def update(self,z):
        # Note: order of Z must be the same as self.z_names
        self.f.x = self.normalise_q(self.f.x)
        self.f.update(z)
        self.f.x = self.normalise_q(self.f.x)
        self.x_history.append(self.f.x)
        self.var_history.append(np.diag(self.f.P))
        self.z_history.append(z)
        self.t += self.dt_data
        self.t_history.append(self.t)
        return self.f.x

    def normalise_q(self,x):
        q_ = np.array([x[self.vehicle.X_map[qi]] for qi in ["q0","q1","q2","q3"]]) 
        q_ = q_ / np.linalg.norm(q_)
        for qi,q_name in zip(q_,["q0","q1","q2","q3"]):
            x[self.vehicle.X_map[q_name]] = qi
        return x

    def plot(self):
        X_KF = np.array(self.x_history).T
        var_KF = np.array(self.var_history).T
        data = pd.DataFrame(np.array(self.z_history),columns=self.z_names)
        time = np.array(self.t_history)
        for ax in self.ax:
            ax.clear()
        self.fig.canvas.flush_events()

        self.ax[0].plot(time,data["x_inertia"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        self.ax[0].plot(time,data["y_inertia"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        self.ax[0].plot(time,data["z_inertia"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        if hasattr(self,"true_data"):
            self.ax[0].plot(self.true_data["time"],self.true_data["x_inertia"],"--",label="true_data",linewidth=1,color='k')
            self.ax[0].plot(self.true_data["time"],self.true_data["y_inertia"],"--",linewidth=1,color='k')
            self.ax[0].plot(self.true_data["time"],self.true_data["z_inertia"],"--",linewidth=1,color='k')
        self.ax[0].plot(time,X_KF[10],label="KF",color='r')
        self.ax[0].plot(time,X_KF[11],color='r')
        self.ax[0].plot(time,X_KF[12],color='r')
        self.ax[0].errorbar(time[-1],X_KF[10][-1],yerr=var_KF[10][-1],color='grey')
        self.ax[0].errorbar(time[-1],X_KF[11][-1],yerr=var_KF[11][-1],color='grey')
        self.ax[0].errorbar(time[-1],X_KF[12][-1],yerr=var_KF[12][-1],color='grey')

        self.ax[0].set_title("Position")
        self.ax[0].set_xlabel("time [s]")
        self.ax[0].set_ylabel("position [m]")

        self.ax[1].plot(time,data["q0"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        self.ax[1].plot(time,data["q1"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        self.ax[1].plot(time,data["q2"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        self.ax[1].plot(time,data["q3"],"o",label="data",markersize=3,color='k',markerfacecolor='None')
        if hasattr(self,"true_data"):
            self.ax[1].plot(self.true_data["time"],self.true_data["q0"],"--",label="self.true_data",linewidth=1,color='k')
            self.ax[1].plot(self.true_data["time"],self.true_data["q1"],"--",linewidth=1,color='k')
            self.ax[1].plot(self.true_data["time"],self.true_data["q2"],"--",linewidth=1,color='k')
            self.ax[1].plot(self.true_data["time"],self.true_data["q3"],"--",linewidth=1,color='k')
        self.ax[1].plot(time,X_KF[6],label="KF",color='r')
        self.ax[1].plot(time,X_KF[7],color='r')
        self.ax[1].plot(time,X_KF[8],color='r')
        self.ax[1].plot(time,X_KF[9],color='r')
        self.ax[1].errorbar(time[-1],X_KF[6][-1],yerr=var_KF[6][-1],color='grey')
        self.ax[1].errorbar(time[-1],X_KF[7][-1],yerr=var_KF[7][-1],color='grey')
        self.ax[1].errorbar(time[-1],X_KF[8][-1],yerr=var_KF[8][-1],color='grey')
        self.ax[1].errorbar(time[-1],X_KF[9][-1],yerr=var_KF[9][-1],color='grey')
        self.ax[1].set_title("Quaternions")
        self.ax[1].set_xlabel("time [s]")
        self.ax[1].set_ylabel("quaternions")
        self.ax[1].set_ylabel("-")

        self.ax[2].plot(time,X_KF[3],label="KF",color='r')
        self.ax[2].plot(time,X_KF[4],color='r')
        self.ax[2].plot(time,X_KF[5],color='r')
        self.ax[2].errorbar(time[-1],X_KF[3][-1],yerr=var_KF[3][-1],color='grey')
        self.ax[2].errorbar(time[-1],X_KF[4][-1],yerr=var_KF[4][-1],color='grey')
        self.ax[2].errorbar(time[-1],X_KF[5][-1],yerr=var_KF[5][-1],color='grey')
        self.ax[2].set_title("Rate of rotation")
        self.ax[2].set_xlabel("time [s]")
        self.ax[2].set_ylabel("Rotation [rad/s]")
        if hasattr(self,"true_data"):
            self.ax[2].plot(self.true_data["time"],self.true_data["p_body"],"--",label="self.true_data",linewidth=1,color='k')
            self.ax[2].plot(self.true_data["time"],self.true_data["q_body"],"--",linewidth=1,color='k')
            self.ax[2].plot(self.true_data["time"],self.true_data["r_body"],"--",linewidth=1,color='k')

        if hasattr(self,"true_data"):
            self.ax[3].plot(self.true_data["time"],self.true_data["u_body"],"--",label="self.true_data",linewidth=1,color='k')
            self.ax[3].plot(self.true_data["time"],self.true_data["v_body"],"--",linewidth=1,color='k')
            self.ax[3].plot(self.true_data["time"],self.true_data["w_body"],"--",linewidth=1,color='k')
        self.ax[3].plot(time,X_KF[0],label="KF",color='r')
        self.ax[3].plot(time,X_KF[1],color='r')
        self.ax[3].plot(time,X_KF[2],color='r')
        self.ax[3].errorbar(time[-1],X_KF[0][-1],yerr=var_KF[0][-1],color='grey')
        self.ax[3].errorbar(time[-1],X_KF[1][-1],yerr=var_KF[1][-1],color='grey')
        self.ax[3].errorbar(time[-1],X_KF[2][-1],yerr=var_KF[2][-1],color='grey')
        self.ax[3].set_title("Velocity")
        self.ax[3].set_xlabel("time [s]")
        self.ax[3].set_ylabel("Velocity body frame [m/2]")


        plt.tight_layout()
        plt.pause(0.001)
        self.fig.canvas.draw()

#!/usr/bin/python3.8
"""RK4 standalone intergrator
This can be used with Kalman filters and other codes, 
when a non-linear state transform function is needed.
It is intended to intergrate with the VEHICLE_MASTER class
"""

def RK4(plant,X_current,dt,time):
    # apply RK4 method
    h = dt
    plant.set_X(X_current, time=time)
    k1 = plant.get_X_dot()
    plant.set_X(X_current+h/2*k1, time=time+h/2)
    k2 = plant.get_X_dot()
    plant.set_X(X_current+h/2*k2, time=time+h/2)
    k3 = plant.get_X_dot()
    plant.set_X(X_current+h*k3, time=time+h)
    k4 = plant.get_X_dot()
    X_new = X_current + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return X_new

