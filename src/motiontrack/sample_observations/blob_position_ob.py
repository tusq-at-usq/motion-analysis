""" 2-dimensional position and quaternion observable, using image blob data
"""

import numpy as np
import PIL
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from motiontrack.observation import Observation
from motiontrack.camera import CameraView
from motiontrack.geometry import BodySTL
from motiontrack.plot import PlotMatch
from motiontrack.features import BlobsFrame

from frame_blob_detection import get_blobs

class BlobPosition(Observation):
    """ Observable which uses one or more X-Y blob location datasets to
    estimate location and rotation.
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
                 plot_switch=True,
                 delay=0):

        name = "blob_track_observable"
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
        self.delay = delay
        self.t_current += self.delay
        self.threshold = threshold
        self.image = None
        self.visible_blobs = np.empty([])
        self.y = np.empty([])
        self.assignment = np.empty([])
        self.distances = np.empty([])

    def _get_next_t(self):
        if self.index >= self.n_frames:
            return np.Inf
        t_next = self.t_current + self.frame_rate
        return t_next

    def update(self, x_pr: np.array, x_dict: dict):
        counter = self.start + (self.index*(1+self.skip))
        im_path = self.prefix + ('%04i' %counter) +'.tif'

        image = np.array(PIL.Image.open(im_path))
        if self.flip:
            image = np.flip(image, axis=1)
        self.image = image

        r_pr = np.array([x_pr[x_dict[_var]] for _var in ["x", "y", "z"]])
        q_pr = np.array([x_pr[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
        q_pr = q_pr/np.linalg.norm(q_pr)

        z_blobs = self.view.project(r_pr, q_pr)[0]
        z = z_blobs.points
        self.visible_blobs = self.view.get_visible_blobs()

        frame = BlobsFrame(*get_blobs(image))
        self.y = frame.points

        dist = distance.cdist(self.y,z)

        bad = np.where(np.min(dist,axis=1)>self.threshold)
        dist = np.delete(dist,bad,axis=0)
        self.y = np.delete(self.y,bad, axis=0)

        self.assignment = linear_sum_assignment(dist)
        self.distances = dist[self.assignment]
        self.change_size(len(self.distances)*2)

        if self.plot_switch:
            self.plot.update_observation(self.y.T)

    def update_plot(self,x_p, x_dict):
        r_pr = np.array([x_p[x_dict[_var]] for _var in ["x", "y", "z"]])
        q_pr = np.array([x_p[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
        q_pr = q_pr/np.linalg.norm(q_pr)
        z_blobs = self.view.project(r_pr, q_pr)[0]
        if self.plot_switch:
            self.plot.update_projection(z_blobs)
            mesh = self.view.get_mesh()
            self.plot.update_mesh(*mesh)
        self.plot.update_image(self.image)

    def _next_measurement(self, x_pr: np.array,
                          x_dict: dict):

        t_next = self.get_next_t()
        self.t_current = t_next

        # We have a list of measurements 'y' (image blobs), and system
        # observations 'z'. We need to match pairs, while discarding likely
        # false observations.

        tau_next = np.full(self.size,9)
        y = np.concatenate(self.y[self.assignment[0]]).flat
        return t_next, y, tau_next

    def _create_ob_fn(self, x_dict: dict, u_dict:dict, x_pr: np.array) -> np.array:

        def hx(X, _):
            r = np.array([X[x_dict[_var]] for _var in ["x", "y", "z"]])
            q = np.array([X[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]])
            q = q / np.linalg.norm(q)
            self.body.update(r,q)
            blobs_xyz = self.body.blobs[self.visible_blobs].T
            _z = self.view._transform(blobs_xyz).T
            return np.concatenate(_z[self.assignment[1]]).flat
        return hx


