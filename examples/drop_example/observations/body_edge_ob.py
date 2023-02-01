""" 2-dimensional position and quaternion observable, using image blob data
"""

import numpy as np
from scipy.spatial import distance
import cv2 as cv

from motiontrack.observation import Observation
from motiontrack.camera import CameraView
from motiontrack.geometry import BodySTL
from motiontrack.plot import PlotMatch

class BodyEdge(Observation):
    """ Observable which uses one or more X-Y blob location datasets to
    estimate location and rotation.


    TODO: To be more accurate, this function should really un-distort an image,
    and then match edges with undistorted edge projections.
    """
    def __init__(self,
                 prefix: str,
                 body: BodySTL,
                 view: CameraView,
                 plot: PlotMatch,
                 frame_rate: float,
                 start: int,
                 threshold: float,
                 n_frames: int,
                 skip: int=1,
                 flip = False,
                 plot_switch=True):

        name = "line_observable"
        ob_names = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        super().__init__(name=name,
                         size=0,
                         ob_names=ob_names)

        self.plot = plot
        self.view = view
        self.body = body
        self.start = start
        self.skip = skip
        self.prefix = prefix
        self.n_frames = n_frames
        self.t_current = 0
        self.frame_rate = frame_rate
        self.flip = flip
        self.plot_switch = plot_switch
        self.threshold=threshold
        self.image = None

    def _get_next_t(self):

        if self.index >= self.n_frames:
            return np.Inf
        t_next = self.t_current + self.frame_rate
        return t_next

    def update(self, x_pr: np.array, x_dict: dict):

        # Plot function to use for debugging/testing settings
        #  def plot(im,line_st, line_en):
            #  plt.figure(figsize=(12,12))
            #  plt.imshow(im)
            #  plt.plot(np.array([line_st[:,0],line_en[:,0]]),
                     #  np.array([line_st[:,1],line_en[:,1]]),
                     #  marker='x')
            #  plt.show(block=False)
            #  plt.pause(0.1)
            #  input("Press to continue")
            #  plt.close()

        counter = self.start + (self.index*(1+self.skip))
        im_path = self.prefix + ('%04i' %counter) +'.tif'
 
        r_pr = np.array([x_pr[x_dict[_var]] for _var in ["x", "y", "z"]])
        q_pr = np.array([x_pr[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
        q_pr = q_pr/np.linalg.norm(q_pr)
        self.body.update(r_pr,q_pr)

        #  image = np.array(PIL.Image.open(im_path))
        image = cv.imread(im_path)
        if self.flip:
            image = np.flip(image, axis=1)
        self.image = image

        r_pr = np.array([x_pr[x_dict[_var]] for _var in ["x", "y", "z"]])
        q_pr = np.array([x_pr[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
        q_pr = q_pr/np.linalg.norm(q_pr)


        #  img_un = cv.undistort(image, self.view.cal.mtx, self.view.cal.dist)
        image = cv.bitwise_not(image)
        img_un = image
        gray = cv.cvtColor(img_un, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3,3), 0)
        #  thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.ADAPTIVE_THRESH_GAUSSIAN_C,151,3)
        edges = cv.Canny(image=blur, threshold1=75, threshold2=150, L2gradient = True, apertureSize=3)
        lines = cv.HoughLinesP(edges, 5, np.pi / 720, 2, minLineLength=60,maxLineGap=10)
        if lines is not None:
            lines = lines.reshape(-8,4)
        else:
            lines = np.array([[],[],[],[]]).T
        #  lines = lines.reshape(-8,4)
        line_st = np.array([lines[:,0], lines[:,1]]).T
        line_en = np.array([lines[:,2], lines[:,3]]).T
        grads = np.arctan2(np.abs((line_en[:,1]-line_st[:,1])),np.abs(line_en[:,0]-line_st[:,0]))
    
        #  mesh = self.view.get_uncorrected_mesh()[0]
        mesh, self.visible_surfs = self.view.get_mesh()
        mesh_dim = mesh.shape[2]
        n_surf = mesh.shape[0]
        mesh = np.concatenate((mesh, mesh[:,0,:].reshape(n_surf,1,mesh_dim)),axis=1)
        mesh = mesh.reshape(-1,mesh_dim)
        start = np.array([True, True, True, False]*n_surf)
        end = np.array([False, True, True, True]*n_surf)
        mesh_st = mesh[start]
        mesh_en = mesh[end]
        mesh_grads = np.arctan2(np.abs((mesh_en[:,1]-mesh_st[:,1])),np.abs(mesh_en[:,0]-mesh_st[:,0]))
        # Check whether gradients are close
        diff = np.abs(np.subtract.outer(grads, mesh_grads))
        matches = np.array(np.where((np.abs(diff))<self.threshold)).T

        mesh_st = mesh_st[matches[:,1],:]
        mesh_en = mesh_en[matches[:,1],:]
        mesh_grads = mesh_grads[matches[:,1]]
        
        line_st = line_st[matches[:,0],:]
        line_en = line_en[matches[:,0],:]
        grads = grads[matches[:,0]]

        # Check lines are near body

        # Check if lines lie near mesh lines
        threshold2 = 10 # TODO: add as init parameter
        p_matches = []
        for i,(m_st, m_en, l_st, l_en) in enumerate(zip(mesh_st, mesh_en, line_st, line_en)):
            mesh_len = np.linalg.norm(m_en-m_st)
            dist = distance.cdist(np.array([m_st,m_en]),np.array([l_st,l_en]))
            if np.max(dist) < mesh_len:
                d1=np.cross(m_en-m_st,l_st-m_st)/ \
                    np.linalg.norm(m_en-m_st)
                d2=np.cross(m_en-m_st,l_en-m_st)/ \
                    np.linalg.norm(m_en-m_st)
                if np.max(np.abs((d1, d2))) < threshold2:
                    p_matches.append(i)

        duplicate_check = line_st[p_matches,:]
        unique_ps = np.unique(duplicate_check, axis=0, return_index=True)[1]
        p_matches = np.array(p_matches, dtype=int)[unique_ps]

        matches = matches[p_matches]
        mesh_st = mesh_st[p_matches,:]
        mesh_en = mesh_en[p_matches,:]
        mesh_grads = mesh_grads[p_matches]

        line_st = line_st[p_matches,:]
        line_en = line_en[p_matches,:]
        grads = grads[p_matches]

        self.matches = matches
        self.line_st = line_st
        self.line_en = line_en
        self.grads = grads
        self.change_size(len(grads))
        #  plot(edges, line_st, line_en)
        #  plot(image, line_st, line_en)

    def update_plot(self, _1, _2):
        self.plot.update_edges(self.line_st, self.line_en)

    def _next_measurement(self, x_pr: np.array,
                          x_dict: dict):

        t_next = self.get_next_t()
        self.t_current = t_next

        tau_next = np.full(self.size,0.001)
        y = self.grads
        return t_next, y, tau_next

    def _create_ob_fn(self, x_dict: dict, u_dict:dict, x_pr: np.array) -> np.array:

        def hx(X, _):
            r = np.array([X[x_dict[_var]] for _var in ["x", "y", "z"]])
            q = np.array([X[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
            q = q / np.linalg.norm(q)
            self.body.update(r,q)
            #  mesh = self.view.get_uncorrected_mesh()[0]
            mesh = self.view.get_mesh(visible_surfs=self.visible_surfs)[0]
            mesh_dim = mesh.shape[2]
            n_surf = mesh.shape[0]
            mesh = np.concatenate((mesh, mesh[:,0,:].reshape(n_surf,1,mesh_dim)),axis=1)
            mesh = mesh.reshape(-1,mesh_dim)
            start = np.array([True, True, True, False]*n_surf)
            end = np.array([False, True, True, True]*n_surf)
            mesh_st = mesh[start]
            mesh_en = mesh[end]
            mesh_grads = np.arctan2(np.abs(mesh_en[:,1]-mesh_st[:,1]),np.abs(mesh_en[:,0]-mesh_st[:,0]))
            matched_grads = mesh_grads[self.matches[:,1]]
            return matched_grads
        return hx

