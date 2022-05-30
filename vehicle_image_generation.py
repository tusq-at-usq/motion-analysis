#!/usr/bin/python3.8
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from getopt import getopt
import sys
import os
from copy import copy

""" 
DEPENDENCIES:
Python package "Pillow" (i.e. pip3 install Pillow)
Pyton package "ffmpeg" (i.e. pip3 install ffmpeg)
Linux package "ffmpeg" (i.e. sudo apt install ffmpeg)

Define a body for using in graphics creation/interpretation of results
Geometry is handled by POINTS, SURFACES (made of points), and BODY (made of surfaces). 
After initialisation, a local coordinate system X,Y,Z and local->body psi,theta,phi are updated at each time step.
Only surfaces which have a positive dot product of surface normal vector and camera position are plotted.

The local level coordinate system is defined as:

    /|\ 1 (north)
     |
     |
     X--------> 2 (east)
  3 (into earth)

The vehicle/body coordinate system is defined as

    X----------> 1 (forward)
    | \
    |   \
    |    _\| 2 (right wing)
    \/
   3 (down)

 
Authors: Andrew Lock
First created: Jan 2022

TODO:
1.Add blobs for curved surfaces, with an associated vector from CoG

"""

def main(uoDict):
    filename = uoDict["--file"]
    if "--save" in uoDict:
        saveSwitch = True
    else:
        saveSwitch = False
    runFromCSV(filename,saveSwitch)

class POINT:
    # Class to define individual points
    def __init__(self,Xi):
        # Initialaisation coordinates
        self.Xi = np.array(Xi)
        # Local coordinates and local frame
        self.XL = np.zeros(3)
        # Body coordinates and body frame (static after initialisation)
        self.XB = np.zeros(3)
        # Body centre in local coordinates and local frame 
        self.XC = np.zeros(3)
        # Rotation matrix between local and body coordinates
        # self.T_rot = np.zeros((3,3)) # TODO: delete if unnecessary

    def initialise(self,body_CS_origin,T_body_CS):
        # Set body coordiantes, using the initialisation frame
        self.XB = self.Xi - body_CS_origin
        self.XB = np.matmul(T_body_CS,self.XB)

    def update(self,C_vec,T_LB):
        self.XC = C_vec
        self.T_LB = T_LB
        self.XL = np.matmul(T_LB,self.XB)+self.XC

class BLOB:
    def __init__(self,Xi):
        # Initialaisation coordinates
        self.Xi = np.array(Xi)
        # Local coordinates and local frame
        self.XL = np.zeros(3)
        # Body coordinates and body frame (static after initialisation)
        self.XB = np.zeros(3)
        # Body centre in local coordinates and local frame 
        self.XC = np.zeros(3)

    def initialise(self,body_CS_origin,T_body_CS):
        # Set body coordiantes, using the initialisation frame
        self.XB = self.Xi - body_CS_origin
        self.XB = np.matmul(T_body_CS,self.XB)

    def update(self,C_vec,T_LB):
        self.XC = C_vec
        self.T_LB = T_LB
        self.XL = np.matmul(T_LB,self.XB)+self.XC

class SURF:
    # A surface is a collection of points on a flat plane, defined counter-clockwise for outwards normal (RHR)
    def __init__(self,points,name,colour):
        self.points = points
        self.blobs = []
        self.name = name
        self.check_plane()
        self.n1 = self.calc_normal("i")
        self.A = self.calc_area()
        self.colour = colour

    def check_plane(self):
        # Check that the points are on a plane using the determinant method
        if len(self.points) > 3:
            temp = np.vstack([np.array([p.Xi for p in self.points]).T,np.ones(len(self.points))])
            det = la.det(temp)
            if not np.isclose(det,0):
                print("WARNING: Surface points on ",self.name,"are not on a common plane")

    def calc_normal(self,frame="L"):
        # Calculate the surface outward normal vector
        if frame == "L":
            BA = self.points[1].XL - self.points[0].XL
            CA = self.points[2].XL - self.points[0].XL
        if frame == "i":
            BA = self.points[1].Xi - self.points[0].Xi
            CA = self.points[2].Xi - self.points[0].Xi
        dirvec = np.cross(BA,CA)
        n = dirvec/la.norm(dirvec)
        self.n = n
        return n

    def calc_area(self):
        # Calculate the surface area by summming the cross-product of edges
        area = 0
        for i in range(len(self.points)):
            area = area + (np.cross(self.points[i-1].Xi,self.points[i].Xi))
        area = la.norm(area)/2
        return area

class BODY:
    # A body is a closed collection of surfaces
    def __init__(self,surfs,name):
        self.surfs = surfs
        self.name = name
        self.check_closed()
        self.points = sum([s.points for s in self.surfs],[])
        self.blobs = sum([s.blobs for s in self.surfs],[])
        self.t=0
        self.blobSize = 1/5

    def check_closed(self):
        # Check the body is closed by summing the product of area and normal vector
        check = la.norm(sum(s.A*s.n1 for s in self.surfs))
        if not np.isclose(check,0):
            print("WARNING: body",self.name," is not closed")
        # else:
            # print("Body",self.name,"is closed")

    def input_centroid(self,XI,psi,theta,phi):
        self.XI = XI
        q0, q1, q2, q3 = self.euler_to_quaternion(psi,theta,phi)
        T_CS = self.calculate_direction_cosine_from_quaternions(q0,q1,q2,q3)
        [p.initialise(XI,T_CS) for p in self.points]
        [b.initialise(XI,T_CS) for b in self.blobs]

    def update(self,X,Q,t):
        self.t = t
        if all(q == 0 for q in Q):
            # Creating T_BL for no translation results in a [0] matrix
            T_BL = np.identity(3)
        else:
            q0,q1,q2,q3 = Q
            # Use Euler angles for now, but later use Quaternion input for integration with system_sim code
            # q0, q1, q2, q3 = self.euler_to_quaternion(psi,theta,phi)
            T_BL = self.calculate_direction_cosine_from_quaternions(q0,q1,q2,q3)
        [p.update(X,T_BL.T) for p in self.points]
        [b.update(X,T_BL.T) for b in self.blobs]
        self.update_normals()

    def update_normals(self):
        for s in self.surfs:
            s.calc_normal()

    def get_points(self):
        return [p.XL for p in self.points]

    def euler_to_quaternion(self,psi, theta, phi):
        """Convert Euler angles to quaternions."""
        # According to Zipfel Eqn. 10.12
        q0 = np.cos(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        q1 = np.cos(psi/2) * np.cos(theta/2) * np.sin(phi/2) \
            - np.sin(psi/2) * np.sin(theta/2) * np.cos(phi/2)
        q2 = np.cos(psi/2) * np.sin(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.cos(theta/2) * np.sin(phi/2)
        q3 = np.sin(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            - np.cos(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        return q0, q1, q2, q3

    def quaternion_to_euler(self,q0, q1, q2, q3):
        """Convert Quternion to Euler angles."""
        # According to Zipfel Eqn. 10.12
        psi = np.arctan(2 * (q1 * q2 + q0 * q3)
            / (q0**2 + q1**2 - q2**2 - q3**2))
        theta = np.arcsin(-2 * (q1 * q3 - q0 * q2))
        phi = np.arctan(2 * (q2 * q3 + q0 * q1)
            / (q0**2 - q1**2 - q2**2 + q3**2))
        # TODO: Need to adjust code to manage singularities at psi = +/- pi/2 and phi = +/- pi/2
        return psi, theta, phi

    def calculate_direction_cosine_from_quaternions(self,q0, q1, q2, q3):
        """Calculate directional cosine matrix from quaternions."""
        T_rot = np.array([
            [q0**2+q1**2-q2**2-q3**2,  2*(q1*q2+q0*q3),  2*(q1*q3-q0*q2) ],
            [2*(q1*q2-q0*q3),  q0**2-q1**2+q2**2-q3**2,  2*(q2*q3+q0*q1) ],
            [2*(q1*q3+q0*q2),  2*(q2*q3-q0*q1),  q0**2-q1**2-q2**2+q3**2 ]
            ]) # Eqn 10.14 from Zipfel
        # T_BL += 0.5* (np.eye(3,3) - T_BI.dot(T_BI.T)).dot(T_BI)
        return T_rot

class VIEW:
    # A projection of the 3D local coordinates to a specific 2D direction

    # TODO - can add a perspective projection option later if needed
    def __init__(self,B,perspective="parallel",viewAngle=[0,0,0],name=None,dirName="default",saveSwitch=False,scale=1):
        self.perspective=perspective
        self.viewAngle = viewAngle
        self.s_LV = [] # Vector from local origin to camera origin in local coordinates
        self.B = B
        self.surfPoints = [] # Updated at each timestep
        self.name = name
        self.dirName = dirName
        self.saveSwitch = saveSwitch
        self.artists = []
        self.blobSize = B.blobSize
        self.scale = scale  #Scale of the view image in pixels/m
        self.initialisation()

    def initialisation(self):
        # Create a rotation matrix between local frame and camera frame
        q0,q1,q2,q3 = self.euler_to_quaternion(self.viewAngle[0],self.viewAngle[1],self.viewAngle[2])
        self.T_VL = self.calculate_direction_cosine_from_quaternions(q0,q1,q2,q3)
        self.s_LV = np.matmul(np.array([0,0,-1]),self.T_VL)

    def initialise_plot(self):
        self.fig,self.ax = plt.subplots()
        self.ax.set_title("Projection:"+self.name)
        self.ax.set_facecolor('grey')
        self.ax.set_aspect('equal')

    def update(self):
        # Update surface and blob 2D coordinates from the BODY class 3D coordinates
        # Determine which surfaces are visible
        # Also determine the blob size (approximated by dot product of surface normal and camera direction)
        self.all_points = []
        self.all_blobs = []
        self.colours = []
        self.labels = []
        self.blob_sizes = []
        for surf in self.B.surfs:
            surfPoints = []
            surfBlobs = []
            if np.dot(self.s_LV,surf.n) > 0:
                points = np.array([p.XL for p in surf.points]+[surf.points[0].XL])
                for point in points:
                    surfPoints.append(np.matmul(self.T_VL,point))
                blobs = np.array([b.XL for b in surf.blobs])
                markersize = self.blobSize * np.dot(self.s_LV,surf.n)
                for blob in blobs:
                    surfBlobs.append(np.matmul(self.T_VL,blob))
                self.all_points.append(np.array(surfPoints))
                self.all_blobs.append(np.array(surfBlobs))
                self.blob_sizes.append(markersize)
                self.colours.append(surf.colour)
                self.labels.append(surf.name)

    def get_2D_data(self):
        Xs = []
        Ys = []
        Ds = []
        for blobs,blobSize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
            # if len(blobs)>0:
                Xs.append(blobs[:,1])
                Ys.append(blobs[:,0])
                D = np.empty((blobs[:,1].size))
                D.fill(blobSize)
                Ds.append(D)
        Xs = np.array(Xs).flatten()
        Ys = np.array(Ys).flatten()
        Ds = np.array(Ds).flatten()
        return Xs,Ys,Ds

    def plot(self):
        if not hasattr(self,"fig"):
            self.initialise_plot()
        # Create a frame of the animation
        ax = self.ax
        fillArtists = []
        for points,colour,label in zip(self.all_points,self.colours,self.labels):
            fill, = ax.fill(points[:,1],points[:,0],color=colour,label=label)
            fillArtists.append(fill)
        blobArtists = []
        for blobs,markersize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                blob, = ax.plot(blobs[:,1],blobs[:,0],'o',color='k',markersize=markersize)
                blobArtists.append(blob)
        self.artists.append(fillArtists + blobArtists)
        return fillArtists + blobArtists

    def plot_show(self):
        # Create a frame of the animation
        fig,ax = plt.subplots()
        fillArtists = []
        for points,colour,label in zip(self.all_points,self.colours,self.labels):
            fill, = ax.fill(points[:,1],points[:,0],color=colour,label=label)
            fillArtists.append(fill)
        blobArtists = []
        for blobs,markersize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                blob, = ax.plot(blobs[:,1],blobs[:,0],'o',color='k',markersize=markersize*20)
                blobArtists.append(blob)
        plt.pause(0.1)
        plt.show()

    def plot_vehicle(self):
        if not hasattr(self,"fig"):
            self.initialise_plot()
        self.ax.clear()
        self.ax.set_title(self.name)
        fillArtists = []
        for points,colour,label in zip(self.all_points,self.colours,self.labels):
            fill, = self.ax.fill(points[:,1],points[:,0],color=colour,label=label)
            points = np.vstack((points,points[0]))
            fill, = self.ax.plot(points[:,1],points[:,0],'-',color='k',label=label)
        for blobs,markersize in zip(np.array(self.all_blobs),self.blob_sizes):
            if len(blobs)>0:
                blob, = self.ax.plot(blobs[:,1],blobs[:,0],'o',color='k',markersize=markersize*10)
        plt.pause(0.01)
        self.fig.canvas.draw()

    def create_animation(self):
        # Compile the animation frames and save, if --save in options
        self.ax.set_facecolor('None')
        aniA = animation.ArtistAnimation(self.fig, self.artists)
        if self.saveSwitch:
            aniA.save(self.name+".avi",bitrate=200,dpi=100)
        return aniA

    def euler_to_quaternion(self,psi, theta, phi):
        """Convert Euler angles to quaternions."""
        # According to Zipfel Eqn. 10.12
        q0 = np.cos(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        q1 = np.cos(psi/2) * np.cos(theta/2) * np.sin(phi/2) \
            - np.sin(psi/2) * np.sin(theta/2) * np.cos(phi/2)
        q2 = np.cos(psi/2) * np.sin(theta/2) * np.cos(phi/2) \
            + np.sin(psi/2) * np.cos(theta/2) * np.sin(phi/2)
        q3 = np.sin(psi/2) * np.cos(theta/2) * np.cos(phi/2) \
            - np.cos(psi/2) * np.sin(theta/2) * np.sin(phi/2)
        return q0, q1, q2, q3

    def calculate_direction_cosine_from_quaternions(self,q0, q1, q2, q3):
        """Calculate directional cosine matrix from quaternions."""
        T_rot = np.array([
            [q0**2+q1**2-q2**2-q3**2,  2*(q1*q2+q0*q3),  2*(q1*q3-q0*q2) ],
            [2*(q1*q2-q0*q3),  q0**2-q1**2+q2**2-q3**2,  2*(q2*q3+q0*q1) ],
            [2*(q1*q3+q0*q2),  2*(q2*q3-q0*q1),  q0**2-q1**2-q2**2+q3**2 ]
            ]) # Eqn 10.14 from Zipfel
        # T_BL += 0.5* (np.eye(3,3) - T_BI.dot(T_BI.T)).dot(T_BI)
        return T_rot

    def rotation_matrix_from_vectors(self,vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

def runFromCSV(filename,saveSwitch):
    # Process data using a CSV file (only option for now)

    dirName = filename.split(".")[0]
    check_directory = os.path.isdir("./"+dirName)
    if not check_directory and saveSwitch:
        os.makedirs(dirName)

    resultData = pd.read_csv(filename,header=0,skiprows=[1,2]) # Read result data
    resultData = resultData.rename(columns=lambda x: x.strip()) # Remove white spaces from header names
    Xs = resultData["x_inertia"]
    Ys = resultData["y_inertia"]
    Zs = resultData["z_inertia"]
    q0s = resultData["q0"]
    q1s = resultData["q1"]
    q2s = resultData["q2"]
    q3s = resultData["q3"]
    ts = resultData["time"]

    scale = np.max(np.abs(np.array([Xs,Ys,Zs])))
    L = scale/5

    B = cubeGen(12)
    B.L = L

    # The default view angle is looking from above
    EA_above = np.array([0,0,0]) # Euler angle rotation for above
    EA_east = np.array([np.pi/2,np.pi/2,0]) # Euler angle rotation for local west view
    EA_north = np.array([np.pi,np.pi/2,0]) # Euler angle rotation for local north view

    viewA = VIEW(B,"parallel",EA_above,"above",dirName,saveSwitch)
    viewE = VIEW(B,"parallel",EA_east,"east",dirName,saveSwitch)
    viewN = VIEW(B,"parallel",EA_north,"north",dirName,saveSwitch)

    for q0,q1,q2,q3,X,Y,Z,t in zip(q0s,q1s,q2s,q3s,Xs,Ys,Zs,ts):

        # Update object location
        B.update([X,Y,Z],[q0,q1,q2,q3],t) # Update point locations

        # Update views
        viewA.update()
        viewE.update()
        viewN.update()

        # Plot animation frame
        viewA.plot()
        viewE.plot()
        viewN.plot()
        # print(q0,q1,q2,q3)
        # viewA.plot_show()

    A = viewA.create_animation()
    E = viewE.create_animation()
    N = viewN.create_animation()
    plt.show()

def addBlob(surf,blobX,blobY):
    # Add a blob to a suface using the surface local coordiantes (2D) instead of local (3D)
    xs1 = surf.points[1].Xi - surf.points[0].Xi
    xs1u = xs1 / la.norm(xs1) # Unit vector direction 1
    xs2u = np.cross(surf.n,xs1u) # Unit vector direction 2
    surf.blobs.append(BLOB((blobX*xs1u)+(blobY*xs2u)+surf.points[0].Xi))

def wedge():
    # An example wedge object
    p0 = POINT([1,0,0])
    p1 = POINT([1,1,0])
    p2 = POINT([0,1,0.25])
    p3 = POINT([0,0,0.25])
    p4 = POINT([0,0,-0.25])
    p5 = POINT([0,1,-0.25])
    points = [p0,p1,p2,p3,p4,p5]

    bottom = SURF([p0,p1,p2,p3],"bottom",'r')
    top = SURF([p0,p4,p5,p1],"top",'b')
    right = SURF([p1,p5,p2],"right","y")
    left = SURF([p0,p3,p4],"left","orange")
    back = SURF([p2,p5,p4,p3],"back","g")
    surfs = [top,bottom,left,right,back]

    B = BODY(surfs,"wedge")
    B.input_centroid([0.333,0.5,0],0,0,0)
    B.update([0,0,0],[0,0,0,0],0)  # Move centroid to 0,0,0
    return B

def cubeGen(L,blob_location_filename=None):
    # An example cube vehicle shape
    p0 = POINT([0,0,0])
    p1 = POINT([0,1,0])
    p2 = POINT([1,1,0])
    p3 = POINT([1,0,0])
    p4 = POINT([0,0,1])
    p5 = POINT([0,1,1])
    p6 = POINT([1,1,1])
    p7 = POINT([1,0,1])
    points = [p0,p1,p2,p3,p4,p5,p6,p7]
    
    # Scale by length
    for p in points:
        p.Xi = p.Xi*L

    top = SURF([p0,p1,p2,p3],"top","w")
    bottom = SURF([p7,p6,p5,p4],"bottom","w")
    front = SURF([p6,p7,p3,p2],"front","w")
    back = SURF([p0,p4,p5,p1],"back)","w")
    left = SURF([p0,p3,p7,p4],"left","w")
    right = SURF([p1,p5,p6,p2],"right","w")

    surfs = [bottom,top,front,back,left,right]

    # Add random dots (for testing purposes)
    if blob_location_filename is None:
        try:
            rands = np.load("blob_locations.npy")
        except:
            print("Creating new blob locations")
            rands = np.random.rand(6,2,4)
            # rands = rands*0.8 + 0.1
            rands = rands * L
            np.save("dot_XYs.npy",rands)
            print("Saved new blob locations as 'dot_XYs.npy'")
    else:
        rands = np.load(blob_location_filename)

    for surf,rand in zip(surfs,rands):
        randXs = rand[0]
        randYs = rand[1]
        for randX,randY in zip(randXs,randYs):
            addBlob(surf,randX,randY)

    B = BODY(surfs,"cube")
    B.input_centroid([0.5*L,0.5*L,0.5*L],0,0,0)
    B.update([0,0,0],[0,0,0,0],0) # Move centroid to 0,0,0
    B.update_normals()
    B.blobSize = L/10
    return B

def print_usage():
    print("Work in progress")
    print("===============")
    print("Currently defaults to unit cube shape")
    print("")
    print("  vehivle_image_generation.py --file=resultFile.csv")
    print("")
    print("Argument:                    Comment:")
    print("------------------------------------------------------------------------")
    print(" --file=               String containing result file name for results file.")
    print(" --save               Save the animation in a series of TIFF files.")
    print("")

def print_instructions():
    print("")
    print("     Tool to create images of trajectory")
    print("   ==============================================")
    print("Need to add instructions:")

short_options = ""
long_options = ["help", "file=", "save"]


if __name__ == '__main__':
    user_options = getopt(sys.argv[1:], short_options, long_options)
    uo_dict = dict(user_options[0])

    if "--instructions" in uo_dict:
        print_instructions()
        sys.exit(1)

    else:
        main(uo_dict)
        print("\n")
        print("SUCCESS.")
     
