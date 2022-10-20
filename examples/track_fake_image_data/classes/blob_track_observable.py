""" 2-dimensional position and quaternion observable, using image blob data
"""

from typing import List
import numpy as np
from motiontrack.custom_types import ObservationGroup
from motiontrack.body_projection import View
from motiontrack.geometry import BodySTL
from motiontrack.blob_data import read_blob_data
from motiontrack.spatial_match import SpatialMatch
from motiontrack.plot import PlotMatch

class BlobTrack6DoF(ObservationGroup):
    """ Observable which uses one or more X-Y blob location datasets to
    estimate location and rotation.
    """
    def __init__(self,
                 filenames: List[str],
                 body: BodySTL,
                 views: List[View],
                 plots: List[PlotMatch],
                 frame_rate: float):
        name = "blob_track_observable"
        ob_names = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        super().__init__(name=name,
                         size=7,
                         ob_names=ob_names)

        self.frame_dict = read_blob_data(filenames, sep=',')
        self.frames = list(self.frame_dict)
        self.time = list(np.array(self.frames) * frame_rate)
        self.matcher = SpatialMatch(body, views)
        self.plots = plots

    def _get_next_t(self):
        t_current = self.get_t()
        if t_current >= self.time[-1]:
            return np.Inf
        t_next = self.time[np.searchsorted(self.time, t_current, side='right')]
        return t_next


    def _next_measurement(self, x_pr: np.array,
                          x_dict: dict):
        # Note that geometry works in mm, whereas dynamic system works in m

        t_next = self.get_next_t()
        next_frame = self.frames[self.time.index(t_next)]

        p_pr = [x_pr[x_dict[_var]]*1000 for _var in ["x", "y", "z"]]
        q_pr = [x_pr[x_dict[_var]] for _var in ["q0", "q1", "q2", "q3"]]
        q_pr = q_pr/np.linalg.norm(q_pr)

        # WARNING: Lazy ordered list. Frames should ideally be referenced by
        # dictionary key.
        frames = list(self.frame_dict[next_frame].values())
        p, q = self.matcher.run_match(frames,
                                      p_pr,
                                      q_pr,
                                      plots=self.plots)
        p /= 1000
       
        y_next = np.concatenate([p,q])
        tau_next = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        return t_next, y_next, tau_next

    def _create_ob_fn(self, x_dict: dict) -> np.array:

        # Normalise quaternions to unit rotation
        # FIXME: Troubleshoot this
        q_inds =  np.array([x_dict[q_x] for q_x in ['q0', 'q1', 'q2', 'q3']])
        def _normalise_q(x):
            q_raw = x[q_inds]
            q_norm = q_raw / np.linalg.norm(q_raw)
            for q_i,q_ind in zip(q_norm,q_inds):
                x[q_ind] = q_i
            return x

        def hx(x):
            z = np.array([x[x_dict[name]] for name in self.ob_names])
            #  z = _normalise_q(z)
            return z
        return hx

