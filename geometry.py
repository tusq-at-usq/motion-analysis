import numpy as np
from numpy import linalg as la

""" GEOMETRY
Define a body for using in graphics creation/interpretation of results
Geometry is handled by Points, Surfaces (made of points), and Body (made of surfaces). 
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
First created: May 2022

TODO: Major - use .stl body data. Work out how to incorporate this into surface

"""

class Point:
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

    def initialise(self,body_CS_origin,T_body_CS):
        # Set body coordiantes, using the initialisation frame
        self.XB = self.Xi - body_CS_origin
        self.XB = np.matmul(T_body_CS,self.XB)

    def update(self,C_vec,T_LB):
        self.XC = C_vec
        self.T_LB = T_LB
        self.XL = np.matmul(T_LB,self.XB)+self.XC

class Blob:
    #TODO: Differentiate between flat and curved surface blob
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

class Surf:
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

class Body:
    # A body is a closed collection of surfaces
    def __init__(self,surfs,name,axis=[]):
        self.surfs = surfs
        self.name = name
        self.check_closed()
        self.points = sum([s.points for s in self.surfs],axis)
        self.blobs = sum([s.blobs for s in self.surfs],[])
        self.t=0
        self.blobSize = 1/5
        self.axis = axis

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
        self.X = X
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

def addBlob(surf,blobX,blobY):
    # Add a blob to a suface using the surface local coordiantes (2D) instead of local (3D)
    xs1 = surf.points[1].Xi - surf.points[0].Xi
    xs1u = xs1 / la.norm(xs1) # Unit vector direction 1
    xs2u = np.cross(surf.n,xs1u) # Unit vector direction 2
    surf.blobs.append(Blob((blobX*xs1u)+(blobY*xs2u)+surf.points[0].Xi))

def wedge():
    # An example wedge object
    p0 = Point([1,0,0])
    p1 = Point([1,1,0])
    p2 = Point([0,1,0.25])
    p3 = Point([0,0,0.25])
    p4 = Point([0,0,-0.25])
    p5 = Point([0,1,-0.25])
    points = [p0,p1,p2,p3,p4,p5]

    bottom = Surf([p0,p1,p2,p3],"bottom",'r')
    top = Surf([p0,p4,p5,p1],"top",'b')
    right = Surf([p1,p5,p2],"right","y")
    left = Surf([p0,p3,p4],"left","orange")
    back = Surf([p2,p5,p4,p3],"back","g")
    surfs = [top,bottom,left,right,back]

    B = Body(surfs,"wedge")
    B.input_centroid([0.333,0.5,0],0,0,0)
    B.update([0,0,0],[0,0,0,0],0)  # Move centroid to 0,0,0
    return B

def pyramid_gen(L,blob_location_filename=None):
    # An example of a rectangular-based pyramid pointing into the flow
    p0 = Point([0,0,0])
    p1 = Point([0,2,0])
    p2 = Point([1,1,0.5])
    p3 = Point([0,0,1])
    p4 = Point([0,2,1])

    pX  = Point([1.333,1,0.5])
    pY  = Point([0.333,2,0.5])
    pZ  = Point([0.333,1,1.5])

    points = [p0,p1,p2,p3,p4,pX,pY,pZ]
    
    # Scale by length
    for p in points:
        p.Xi = p.Xi*L

    top = Surf([p0,p1,p2],"top","r")
    bottom = Surf([p3,p2,p4],"bottom","b")
    back = Surf([p0,p3,p4,p1],"back)","orange")
    left = Surf([p0,p2,p3],"left","k")
    right = Surf([p1,p4,p2],"right","w")

    surfs = [bottom,top,back,left,right]

    axis = [pX,pY,pZ]

    # No blobs added yet
    #  if blob_location_filename is None:
        #  try:
            #  # Load the default filename (to avoid accidentallly overwriting)
            #  blob_coords = np.load("blob_XYs.npy")
            #  print("Loaded existing file 'blob_XYs.npy'")
        #  except:
            #  blob_coords = np.random.rand(6,2,4)
            #  # rands = rands*0.8 + 0.1
            #  blob_coords = blob_coords * L
            #  np.save("blob_XYs.npy",blob_coords)
            #  print("Saved new blob locations as 'blob_XYs.npy'")
    #  else:
        #  blob_coords = np.load(blob_location_filename)

    #  for surf,coord in zip(surfs,blob_coords):
        #  blobXs = coord[0]
        #  blobYs = coord[1]
        #  for blobX,blobY in zip(blobXs,blobYs):
            #  addBlob(surf,blobX,blobY)

    B = Body(surfs,"cube",axis)
    B.input_centroid([0.333*L,1*L,0.5*L],0,0,0)
    B.update([0,0,0],[0,0,0,0],0) # Move centroid to 0,0,0
    B.update_normals()
    B.blobSize = L/10
    return B

def cubeGen(L,blob_location_filename=None):
    # An example cube vehicle shape
    p0 = Point([0,0,0])
    p1 = Point([0,2,0])
    p2 = Point([1,2,0])
    p3 = Point([2,0,0])
    p4 = Point([0,0,1])
    p5 = Point([0,2,1])
    p6 = Point([1,2,1])
    p7 = Point([2,0,1])

    #  pX  = Point([1.333,1,0.5])
    #  pY  = Point([0.333,2,0.5])
    #  pZ  = Point([0.333,1,1.5])
    # TODO: Fix for cube

    points = [p0,p1,p2,p3,p4,p5,p6,p7,pX,pY,pZ]
    
    # Scale by length
    for p in points:
        p.Xi = p.Xi*L

    top = Surf([p0,p1,p2,p3],"top","r")
    bottom = Surf([p7,p6,p5,p4],"bottom","b")
    front = Surf([p6,p7,p3,p2],"front","g")
    back = Surf([p0,p4,p5,p1],"back)","orange")
    left = Surf([p0,p3,p7,p4],"left","k")
    right = Surf([p1,p5,p6,p2],"right","w")

    surfs = [bottom,top,front,back,left,right]

    # Add random dots (for testing purposes)
    if blob_location_filename is None:
        try:
            # Load the default filename (to avoid accidentallly overwriting)
            blob_coords = np.load("blob_XYs.npy")
            print("Loaded existing file 'blob_XYs.npy'")
        except:
            blob_coords = np.random.rand(6,2,4)
            # rands = rands*0.8 + 0.1
            blob_coords = blob_coords * L
            np.save("blob_XYs.npy",blob_coords)
            print("Saved new blob locations as 'blob_XYs.npy'")
    else:
        blob_coords = np.load(blob_location_filename)

    for surf,coord in zip(surfs,blob_coords):
        blobXs = coord[0]
        blobYs = coord[1]
        for blobX,blobY in zip(blobXs,blobYs):
            addBlob(surf,blobX,blobY)

    B = Body(surfs,"cube",axis)
    B.input_centroid([0.5*L,0.5*L,0.5*L],0,0,0)
    B.update([0,0,0],[0,0,0,0],0) # Move centroid to 0,0,0
    B.update_normals()
    B.blobSize = L/10
    return B
