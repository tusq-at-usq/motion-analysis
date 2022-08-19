""" 2-dimensional position and quaternion observable, which reads from a CSV file
"""

from typing import List
import numpy as np
from motiontrack.custom_types import ObservationGroup

class Pos2DandQ(ObservationGroup):
    """ 2-dimensional position observations using CSV file data for testing
    """
    def __init__(self, name: str,
                 ob_names: List[str],
                 t_filename: str,
                 data_filename: str):
        super().__init__(name=name,
                         size=6,
                         ob_names=ob_names)

        self.time = np.loadtxt(t_filename, delimiter=',')
        self.data = np.loadtxt(data_filename, delimiter=',')
        self.t_end = np.max(self.time)

    def _next_measurement(self, x_pr: np.array, x_dict: dict):
        t_current = self.get_t()
        index_next = np.argmax(self.time>t_current)
        t_next = self.time[index_next]
        y_next = self.data[index_next,:]
        tau_next = np.array([2, 2, 0.1, 0.1, 0.1, 0.1])
        return t_next, y_next, tau_next

    def _get_next_t(self):
        t_current = self.get_t()
        if t_current >= self.t_end:
            return np.Inf
        t_next = self.time[np.searchsorted(self.time, t_current, side='right')]
        return t_next

    def _create_ob_fn(self, x_dict: dict) -> np.array:
        def hx(x):
            z = np.array([x[x_dict[name]] for name in self.ob_names])
            return z
        return hx

