import numpy as np
import pickle

from motiontrack.dynamic_system import SystemCont

file = 'data/GHAME3'
with open(file,'rb') as file:
    sys_data = pickle.load(file)

S = SystemCont(sys_data.X,sys_data.P,sys_data.f,sys_data.sym_dict,name='test')
S.initialise()
x1 = S.int_RK4(S.x0,0.01)








