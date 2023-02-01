""" 2-dimensional position and quaternion observable, using image blob data
"""

from typing import List
import numpy as np
from motiontrack.observation import Observation
from motiontrack.camera import CameraView
from motiontrack.geometry import BodySTL
from motiontrack.features import read_blob_data, BlobsFrame, SpatialMatch
from motiontrack.plot import PlotMatch
import PIL

from frame_blob_detection import get_blobs

class BlobTrack6DoF(Observation):
    """ Observable which uses one or more X-Y blob location datasets to
    estimate location and rotation.
    """
    def __init__(self,
                 prefixes: List[str],
                 body: BodySTL,
                 views: List[CameraView],
                 plots: List[PlotMatch],
                 frame_rate: float=0.0001,
                 skips: List[int]=[0],
                 starts: List[int]=[1],
                 n_frames = 100):

        name = "blob_track_observable"
        ob_names = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        super().__init__(name=name,
                         size=7,
                         ob_names=ob_names)

        #  self.frame_dict = read_blob_data(filenames, sep=',')
        #  self.frames = list(self.frame_dict)
        self.matcher = SpatialMatch(body, views)
        self.plots = plots
        self.starts = starts
        self.skips = skips
        self.prefixes = prefixes
        self.n_frames = n_frames
        self.t_current = 0
        self.frame_rate = frame_rate

    def _get_next_t(self):
        t_current = self.get_t()
        if self.index >= self.n_frames:
            return np.Inf
        t_next = self.t_current + self.frame_rate
        #  t_next = self.time[np.searchsorted(self.time, t_current, side='right')]
        return t_next


    def _next_measurement(self, x_pr: np.array,
                          x_dict: dict):

        t_next = self.get_next_t()
        self.t_current = t_next
        counters = [start + (self.index*(1+skip)) for start, skip in zip(self.starts, self.skips)]
        im_paths = [prefix + ('%04i' % counter) +'.tif' for counter,prefix in zip(counters,self.prefixes)]

        ims = [np.array(PIL.Image.open(path)) for path in im_paths]
        ims[1] = np.flip(ims[1], axis=1)

        frames = [BlobsFrame(*get_blobs(im)) for im in ims]

        p_pr = [x_pr[x_dict[_var]] for _var in ["x", "y", "z"]]
        q_pr = [x_pr[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]]
        q_pr = q_pr/np.linalg.norm(q_pr)


        self.plots[0].update_image(ims[0])
        self.plots[1].update_image(ims[1])
        #  self.plots[1].update_image(np.flip(np.array(ims[1])))
        #  self.plots[1].update_image((np.array(ims[1])))


        # WARNING: Lazy ordered list. Frames should ideally be referenced by
        # dictionary key.
        p, q = self.matcher.run_match(frames,
                                      p_pr,
                                      q_pr,
                                      plots=self.plots)
      
        y_next = np.concatenate([p,q])
        tau_next = np.array([0.002, 0.002, 0.002, 0.01, 0.01, 0.01, 0.01])
        return t_next, y_next, tau_next

    def _create_ob_fn(self, x_dict: dict, u_dict:dict) -> np.array:

        # Normalise quaternions to unit rotation
        # FIXME: Troubleshoot this
        q_inds =  np.array([x_dict[q_x] for q_x in ['q0', 'q1', 'q2', 'q3']])
        def _normalise_q(x):
            q_raw = x[q_inds]
            q_norm = q_raw / np.linalg.norm(q_raw)
            for q_i,q_ind in zip(q_norm,q_inds):
                x[q_ind] = q_i
            return x

        def hx(x, u):
            z = np.array([x[x_dict[name]] for name in self.ob_names])
            return z
        return hx

