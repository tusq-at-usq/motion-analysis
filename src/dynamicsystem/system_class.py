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
import pickle
from typing import List, Tuple
import numpy as np
import yaml
import sympy as sp

class DynamicSystem:
    def __init__(self,
                 X: List[sp.Symbol],
                 P: List[sp.Symbol],
                 f: List[sp.Expr],
                 sym_dict: dict,
                 name: str='default_name'):

        self.X = list(X)
        self.P = list(P)
        self.f = list(f)
        self.U = []
        self.J = self._create_jacobian_fn()
        self.sym_dict = sym_dict
        self.name = name

        self.state_dict = {k: v for k, v in self.sym_dict.items() if k in X}
        self.param_dict = {k: v for k, v in self.sym_dict.items() if k in P}
        self.sym_dict_inv = {v: k for k, v in sym_dict.items()}
        self.state_dict_inv = {v: k for k, v in self.state_dict.items()}
        self.param_dict_inv = {v: k for k, v in self.param_dict.items()}

        self.x_dict = {name:i for i,name in enumerate(self.state_dict.values())}

        self.f_np = []

    def get_nx(self):
        return len(self.X)

    def save(self, name: str):
        """
        Save system

        Save dynamic system using pickle

        Parameters
        ----------
        name : string
            Name of the saved file
        """
        with open(name,'wb') as file:
            pickle.dump(self,file)

    def write_default_config(self, config_name: str):
        """/docum
        Write default configuration file

        Default configuration file is populated with default values.
        Asks for an input in the case that 'config_name.yaml' already exists.

        Parameters
        ----------
        config_name : str
            Name of the configuration file ('.yaml' is appended automatically)
        """
        if os.path.isfile(config_name+'.yaml'):
            print ("WARNING: Input file already exists")
            rep = input("Override? (Y/N)")
            if rep not in ['Y','y']:
                print("Not overwriting")
                return
        X_def_dict = {}
        P_def_dict = {}
        for X_i in self.X:
            X_def_dict[self.sym_dict[X_i]] = {'Min':-1.,'Max':1.,'X0':1.}
        for P_i in self.P:
            P_def_dict[self.sym_dict[P_i]] = {'Type':'Param','X0':1.,'Min':'-','Max':'-'}
        default_dict = {'System-name':self.name,'States':X_def_dict,'Parameters':P_def_dict}

        with open(config_name+'.yaml', 'w') as file:
            yaml.dump(default_dict, file, sort_keys=False)
            return

    def load_x_0(self, config_name: str) -> np.array:
        """
        Load state initial values

        Parameters
        ----------
        config_name : str
            Configuration filename ('.yaml' is automatically appended)

        Returns
        -------
        x_0 : np.array
            Initial state vector
        """
        with open(config_name+'.yaml') as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)
        if in_dict['System-name'] != self.name:
            print('WARNING: Config system does not match dynamic system')
        x_0 = [x_i["X0"] for x_i in in_dict['States'].values()]
        return np.array(x_0)

    def load_config(self, config_name: str) -> Tuple[dict, dict, dict, dict]:
        """
        Load system configuration file

        Parameters
        ----------
        config_name : str
            Configuration name

        Returns
        -------
        states : dict
        params : dict
        inputs : dict
        aug_states : dict
        """
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

    def add_aug_states(self, config_name: str):
        """
        Add augmented states from config

        Adds augmented states to the dynamic system from the config file
        for use with Koopman linearised systems.
        Augmented states are indicated by 'Aug_state' type in the config file'

        Parameters
        ----------
        config_name : str
            Name of config file ('.yaml' is automatically appended)
        """
        _, _, _, aug_states = self.load_config(config_name)
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

    def sub_params(self, config_name: str):
        """
        Substitute parameter values into dynamic equations.
        Also separates time-varying inputs into a list and dictionary.

        Parameters
        ----------
        config_name : str
            Name of config file ('.yaml' is automatically appended)
        """
        _, params, inputs, _ = self.load_config(config_name)
        param_subs = [(self.sym_dict_inv[P[0]],P[1]) for P in params.items()]
        self.U = [self.sym_dict_inv[ui] for ui in inputs]
        for i in range(len(self.f)):
            self.f[i] = self.f[i].subs(param_subs)
        return

    def lambdify(self):
        """
        Createy numpy functions for each dynamic function.

        Should only be performed when parameters have been substituted into
        dynamic equations by calling the function `sub_params()`
        Each dynamic function takes a state vector input.
        """
        for f in self.f:
            self.f_np.append(sp.lambdify([self.X, self.U], f,'numpy'))
        self.J_np = sp.lambdify([self.X, self.U], self.J,'numpy')

    def get_x_dot(self, x: np.array, u: np.array=np.array([])):
        """
        Get dx(t)/dt

        Parameters
        ----------
        x : np.array
            State vector
        u : np.array
            Input vector
        """
        x_dot = np.array([f(x, u) for f in self.f_np])
        return x_dot

    def step(self, dt: float, x: np.array, u: np.array=np.array([])):
        """
        Integration time-step dynamic function

        RK4 integrator with constant input over the timestep.

        Parameters
        ----------
        dt : float
            Integration time step
        x : np.array
            State vector
        u : np.array
            Input vector
        """
        k1 = self.get_x_dot(x, u)
        k2 = self.get_x_dot(x+(dt/2)*k1, u)
        k3 = self.get_x_dot(x+(dt/2)*k2, u)
        k4 = self.get_x_dot(x+dt*k3, u)
        x_ = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return x_

    def _create_jacobian(self):
        J = sp.Matrix(self.f).jacobian(self.X+self.U)
        return J

    def integrate(self,
                  t_end: float,
                  x_0: np.array,
                  u_funs: list = [],
                  dt_max:float = 0.01) -> Tuple[np.array, np.array]:
        """
        Time-integrate function

        Uses RK4 integrator.
        Inputs are passed as list of functions [u_1(t), u_2(t), ...].

        Parameters
        ----------
        t_end : float
            End-time of the integration
        x_0 : np.array
            Initial state vector
        u_funs : list
            List of functions u(t) which return the input value for each time
        dt_max : float
            Maximum integration time-step

        Returns
        -------

        Tuple[np.array, np.array]
            [TODO:description]
        """
        if t_end == 0:
            return np.array([x_0]), np.array(0)
        steps = int(np.ceil(t_end/dt_max))
        dt = t_end/steps
        x = x_0
        x_history = [x]
        t = 0
        t_history = [t]
        for _ in range(steps):
            u = np.array([u_i(t) for u_i in u_funs])
            x = self.step(dt, x, u)
            x_history.append(x)
            t = t + dt
            t_history.append(t)
        return np.array(x_history), np.array(t_history)

        #TODO: Input function to load input functions/data





