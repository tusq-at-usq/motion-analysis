""" System class

A generic class which can be used for dynamical system analysis. 

Takes a Sympy-Sysopt backend, and turns it into generic parameters:
X = states
P = parameters (these include fixed parameters and inputs)
f = the dynamical equations f(x,P)
Y = observable variables [optional]
g = observable functions g(x,P) [optional]
"""

import os
import numpy as np
import yaml
import sympy as sp
import pickle

class DynamicSystem:
    def __init__(self,X,P,f,sym_dict,name='default_name'):
        
        self.X = list(X)
        self.P = list(P)
        self.f = list(f)
        self.U = []
        self.sym_dict = sym_dict
        self.name = name

        self.state_dict = {k: v for k, v in self.sym_dict.items() if v in X}
        self.param_dict = {k: v for k, v in self.sym_dict.items() if v in P}
        self.sym_dict_inv = {v: k for k, v in sym_dict.items()}
        self.state_dict_inv = {v: k for k, v in self.state_dict.items()}
        self.param_dict_inv = {v: k for k, v in self.param_dict.items()}

        self.f_np = []

    def save(self,name):
        file = open(name, 'wb')
        pickle.dump(self,file)
        file.close()

    def write_default_config(self,config_name):
        # Warn before overwriting existing config file
        if os.path.isfile(config_name+'.yaml'):
            print ("WARNING: Input file already exists")
            rep = input("Override? (Y/N)")
            if rep not in ['Y','y']:
                print("Not overwriting")
                return

        X_def_dict = {}
        P_def_dict = {}
        for Xi in self.X:
            X_def_dict[self.sym_dict[Xi]] = {'Min':-1.,'Max':1.,'X0':1.}
        for Pi in self.P:
            P_def_dict[self.sym_dict[Pi]] = {'Type':'Param','X0':1.,'Min':'-','Max':'-'}
        default_dict = {'System-name':self.name,'States':X_def_dict,'Parameters':P_def_dict}

        with open(config_name+'.yaml', 'w') as file:
            documents = yaml.dump(default_dict, file, sort_keys=False)
            return

    def load_X0(self,config_name):
        with open(config_name+'.yaml') as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)
        if in_dict['System-name'] != self.name:
            print('WARNING: Config system does not match dynamic system')
        X0 = [Xi["X0"] for Xi in in_dict['States'].values()]
        return np.array(X0)

    def load_config(self,config_name):
        with open(config_name+'.yaml') as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)
        if in_dict['System-name'] != self.name:
            print('WARNING: Config system does not match dynamic system')
        # Augmented states are used for Koopman operator creation
        inputs = {}
        params = {}
        states = in_dict["States"]
        aug_states = {}
        for P in in_dict["Parameters"].items():
            if P[1]["Type"] == 'Param':
                params[P[0]] = P[1]["X0"]
            elif P[1]["Type"] == 'Input':
                inputs[P[0]] = P[1]
            elif P[1]["Type"] == 'Aug_state':
                aug_states[P[0]] = P[1]
        return states, params, inputs, aug_states

    def add_aug_states(self,config_name):
        states,params,inputs,aug_states = self.load_config(config_name)
        aug_states_sub = []
        for i in range(len(aug_states)):
            x_ = sp.Symbol("x"+str(len(self.X)+i))
            self.state_dict[x_] = list(aug_states.keys())[i]
            aug_states_sub.append((self.sym_dict_inv[list(aug_states.keys())[i]],x_))
            self.X.append(x_)
        for i in range(len(aug_states)):
            self.f.append(sp.core.mul.Mul(0.))
        #Re-name augmented states 'x<i>' in dynamics equations
        for i in range(len(self.f)):
            self.f[i] = self.f[i].subs(aug_states_sub)
        return

    def sub_params(self,config_name):
        states, params, inputs, aug_states = self.load_config(config_name)
        param_subs = [(self.sym_dict_inv[P[0]],P[1]) for P in params.items()]
        self.U = [self.sym_dict_inv[ui] for ui in inputs.keys()]
        for i in range(len(self.f)):
            self.f[i] = self.f[i].subs(param_subs)
        return

    def lambdify(self):
        variables = self.X + self.U
        for f in self.f:
            self.f_np.append(sp.lambdify(variables,f,'numpy'))
        return

    def get_xdot(self,X,U=[]):
        XU = list(X)+U
        Xdot = np.array([f(*XU) for f in self.f_np])
        return Xdot

    def step(self,dt,X,U=[]):
        k1 = self.get_xdot(X,U)
        k2 = self.get_xdot(X+(dt/2)*k1,U)
        k3 = self.get_xdot(X+(dt/2)*k2,U)
        k4 = self.get_xdot(X+dt*k3,U)
        X_ = X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return X_

    def integrate(self,t_end,dt,X0,U=[]):
        steps = int(t_end/dt)
        X = X0
        t = 0
        for i in range(steps):
            if not U==[]:
                U = [Ui(t) for Ui in U]
            X = self.step(dt,X,U)
            t = t + dt
        return X








