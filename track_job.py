"""Vehicle tracker job file
"""

import pandas as pd

from idmoc.vehicle_master.vehicle_master import VEHICLE_FLAT_EARTH_6DOF
from idmoc.vehicle_master.vehicle_gravity import GRAVITY_CONSTANT
from idmoc.vehicle_master.vehicle_atmosphere import ATMOSPHERE_CONSTANT
from idmoc.vehicle_master.vehicle_aero import AERODYNAMICS_SPHERE, AERODYNAMICS_LUT_GHAME6, AERODYNAMICS_NONE,AERODYNAMICS_LUT_GHAME6_APPROX

from idmoc.motiontracker.UKF6DoF import *
# from UKF import *

# 0. Basic inputs --------------------------------------------------------
T.result_filename = "tracker_6DoF_results.csv"
T.dt_image = 0.005 # Time between images
T.dt_int_max = 0.0002 # Maximum dynamic integration timestep

# 1. Create vehicle system -----------------------------------------------
# gravity_model = GRAVITY_CONSTANT()
# atmosphere_model = ATMOSPHERE_CONSTANT(rho=1.225, P=1.0135e5, T=300., a=344., mu=1.86e-6)
# aero_model = AERODYNAMICS_SPHERE(forces_frame_type='wind')
# D = 42.67e-3
# area = D**2/4*np.pi
# aero_model.set_geometry_parameters(area=area, L_char=D)
# vehicle = VEHICLE_FLAT_EARTH_6DOF(aero_class=aero_model, atmosphere_class=atmosphere_model, gravity_class=gravity_model)
# vehicle.set_properties(mass=45.93e-3)
# vehicle.set_initial_conditions(u_body=10., v_body=5,  p_body=2, q_body=1, r_body=3, elevation=np.pi/4)
gravity_model = GRAVITY_CONSTANT()
atmosphere_model = ATMOSPHERE_CONSTANT(rho=1.225, P=1.0135e5, T=300., a=344., mu=1.86e-6)
aero_model = AERODYNAMICS_LUT_GHAME6(forces_frame_type='wind', LUT_file='GHAME6_aerotables.py')
# aero_model = AERODYNAMICS_LUT_GHAME6_APPROX(forces_frame_type='wind', LUT_file='GHAME6_aerotables.py')
REFA = 557.42  # m^2
REFB = 24.38  # m
REFC = 22.86  # m
aero_model.set_geometry_parameters(area=REFA, span=REFB, length=REFC)
vehicle = VEHICLE_FLAT_EARTH_6DOF(aero_class=aero_model, atmosphere_class=atmosphere_model, gravity_class=gravity_model)
vehicle.set_properties(mass=10e3, I11=1.e3, I2=1.e3, I33=1.e3, I13=0.)
vehicle.set_initial_conditions(u_body=100.,v_body=200, w_body=30, p_body=0, q_body=0, r_body=0., elevation=np.deg2rad(0.),heading=np.deg2rad(0))

T.vehicle = vehicle

#2. Create vehicle projection --------------------------------------------
T.B = cubeGen(12,"dot_XYs.npy")

# 3. Define true data (optional) -----------------------------------------
true_filename = "GHAME6_1_results.csv"
true_data = pd.read_csv(true_filename,header=0,skiprows=[1,2]) 
T.true_data = true_data.rename(columns=lambda x: x.strip()) # Remove white spaces from header names

# 4. Define the filter, initial state, and optional parameters -----------
T.filter = UKF6DoF(T.vehicle,dt_data=T.dt_image,dt_int_max=T.dt_int_max,
                   true_data=T.true_data)
T.filter.t=3*T.dt_image

P = np.eye(13)*100 # Initial covariance matrix
P[6:10,6:10] = np.eye(4)*0.4
P[3:6,3:6] = np.eye(3)*100
P[0:3,0:3] = np.eye(3)*1000
R = np.zeros((7,7)) # Data variance 
R[0:4,0:4] = np.eye(4)*0.0001
R[4:7,4:7] = np.eye(3)*0.1
# Q = np.ones((13,13))*0.00001 #Process variance 
# Q = np.zeros((13,13)) #Process variance 
Q = np.ones((13,13))*0.0001
# Q = np.eye(13)*0.0000000001
Q[0:3,0:3] = np.eye(3)*0.5
Q[3:6,3:6] = np.eye(3)*0.5
# Q[6:10,6:10] = np.eye(4)*0.00001

XYZ0 = [0,0,0]
Q0 = [1.0,0.0,0.0,0.0]
T.x0 = [150,150,20,0,0,0] + list(Q0) + list(XYZ0)

T.filter.initialise(x0=T.x0,P=P,Q=Q,R=R)

# 5. Define data files and views -----------------------------------------
T.data_files = ["disturbed_tracking_dataA.txt","disturbed_tracking_dataE.txt"]
# T.data_files = ["tracking_dataA.txt","tracking_dataE.txt"]
EA_above = np.array([0,0,0]) # Euler angle rotation for above
EA_east = np.array([np.pi/2,np.pi/2,0]) # Euler angle rotation for local west view
viewA = VIEW(T.B,"parallel",EA_above,"above",scale=18.6257)
viewE = VIEW(T.B,"parallel",EA_east,"east",scale=30.485)
T.views = [viewA,viewE]

# 6. Define the image offsets (optional - otherwise will be calculated) --
# OS1 = np.array([-54.8865,  23.878, 0.0])
# # OS2 = np.array([-49.0400,  0,  18.09387])
# OS2 = np.array([-47.97,  0,  18.09387])
# T.offsets = np.array([OS1,OS2])
T.calculate_offsets = True

