import numpy as np
import pickle
from dynamicsystem.system_class import DynamicSystem

file = 'DoF3'
with open(file,'rb') as file:
    S = pickle.load(file)

S.write_default_config('test')
S.load_config('test')
S.add_aug_states('test')
S.sub_params('test')
S.lambdify()
X0 = S.load_X0('test')
X_ = S.integrate(4,0.1,X0)
print("Integrated state:",X_)
print("SUCCESS")








