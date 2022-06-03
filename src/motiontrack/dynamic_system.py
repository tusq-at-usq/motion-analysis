import os
import numpy as np
import sympy as sp
from bidict import bidict
import yaml

class SystemCont:
    # A container for a continuous-time dynamic system
    # Set up as an interface for systems from Sysopt
    
    def __init__(self,x,P,fx,name_dict,u=[],u_dict={},name='unnamed_system'):
        # The input variables are numpy symbols and expressions.
        self.name = name
        self.x = list(x) # States
        self.P = list(P) # Parameters
        self.f_xuP = fx # Differential equations f(x,u,P)
        self.u = list(u)
        self.u_dict = u_dict # Input function dict {ui,fi(t)}
        self.x0 = {}
        self.P0 = {}

        all_vars = set([*x,*P,*u])
        # Check that variables in eqs are in X,u,P.
        # If not, add to P
        for f in self.f_xuP:
            free_vars = set(f.free_symbols)-all_vars
            [self.P.append(v) for v in list(free_vars)]
            all_vars = all_vars | free_vars

        # Clean up name_dict by only using X,U,P values
        filtered_dict = {k:v for (k,v) in name_dict.items() if k in all_vars}
        self.name_dict = bidict(filtered_dict)

        # Write config file if one doesn't already exist
        if not os.path.isfile(self.name+'.yaml'):
            self.write_default_input()
            print("Saved default input to: "+self.name+".yaml")

    def initialise(self):
        # Substitude parameters and lambdify equations
        f_xu, self.x0 = self.read_config()
        f_np = []
        u = list(self.u_dict.keys())
        for f in f_xu:
            f_np.append(sp.lambdify((self.x,*u),f,'numpy'))
        self.f_np = f_np

    def get_xdot(self,x,t):
        u_ = [f(t) for f in self.u_dict.values()]
        xdot = np.array([f((*x,*u_)) for f in self.f_np])
        return xdot

    def int_RK4(self,x,dt,t=0):
        # apply RK4 method
        k1 = self.get_xdot(x,t)
        k2 = self.get_xdot(x+(dt/2)*k1,t+(dt/2))
        k3 = self.get_xdot(x+(dt/2)*k2,t+(dt/2))
        k4 = self.get_xdot(x+dt*k3,t+dt)
        x_new = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return x_new

    def write_default_input(self):
        # Write a configuration file with parameters and initial values
        x_dict = {}
        P_dict = {}
        for xi in self.x:
            x_dict[self.name_dict[xi]] = {'x0':1.}
        for Pi in self.P:
            P_dict[self.name_dict[Pi]] = {'Val':1.}
        default_dict = {'Sim_name':self.name,'States':x_dict,'Parameters':P_dict}
        with open(self.name+'.yaml', 'w') as file:
            documents = yaml.dump(default_dict, file, sort_keys=False)

    def read_config(self):
        with open(self.name+'.yaml') as file:
            in_dict = yaml.load(file, Loader=yaml.FullLoader)
        # Separate states and constants
        states = in_dict["States"]

        P_sub = []
        for P in in_dict["Parameters"].items():
            P_sub.append((self.name_dict.inv[P[0]],float(P[1]['Val'])))
        f_xu = self.f_xuP.subs(P_sub)

        x0 = []
        for x in in_dict["States"].items():
            x0.append(float(x[1]['x0']))
        return f_xu, np.array(x0)


