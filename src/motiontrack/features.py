
""" Detection and maniupulation of optically detected features

Note: This module is not relevant anymore, and kept primarily for future reference.
"""

from typing import List
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class BlobsFrame:
    """Class to represent 2D blob data found by image recognition
    TODO: better structure for this using dictionaries.
    """

    def __init__(self, points: np.array, diameters: np.array):
        self.points = points  # x,y coordinates
        self.diameters = diameters  # Diameters
        self.n = points.shape[0]

    def remove_blob(self, i: int):
        """
        Remove blob point from frame

        Parameters
        ----------
        i : int
            index of blob to remove
        """
        self.points = np.delete(self.points, i, axis=1)
        self.diameters = np.delete(self.diameters, i)
        self.n -= 1

    def plot_frame(self):
        """
        Plot the 2D blob frame

        """
        fig, ax = plt.subplots()
        ax.plot(self.points[0], self.points[1], "o")
        plt.show()


class BlobFile:
    """Stores all blob data from a file and processes it into frames"""

    def __init__(
        self,
        filename: str,
        frame_key="Frame",
        x_key="x_cord",
        y_key="y_cord",
        size_key="Size",
        sep=" ",
    ):
        blob_data = pd.read_csv(filename, header=0, sep=sep, index_col=False)

        # Find the names of headers for the important parameters (search for part of string)
        # TODO: Add more blob details if we have them
        self.frame_header = list(filter(lambda x: frame_key in x, blob_data.keys()))[0]
        self.x_header = list(filter(lambda x: x_key in x, blob_data.keys()))[0]
        self.y_header = list(filter(lambda x: y_key in x, blob_data.keys()))[0]
        self.size_header = list(filter(lambda x: size_key in x, blob_data.keys()))[0]

        self.name = filename.split("/")[-1].split(".")[0]
        self.data = blob_data
        self.frames = blob_data[self.frame_header]
        self.X = blob_data[self.x_header]
        self.Y = blob_data[self.y_header]
        self.D = blob_data[self.size_header]
        self.frame_start = np.min(blob_data[self.frame_header])
        self.frame_end = np.max(blob_data[self.frame_header])


def read_blob_data(filenames: List[str], sep=" ") -> dict:
    """
    Reads one or more files containing 2D blob data, and produces a dictionary of
    BlobsFrames

    A BlobFrame instance is created for each video frame.
    All frames are stored within a BlobData object.

    Parameters
    ----------
    filenames : List[str]
        A list of filenames containing the blob data

    Returns
    -------
    dict
        A dictionary with structure:
        { <frame no.> : {<filename> : <BlobsFrame> } }

    """

    blob_files = []
    for filename in filenames:
        blob_files.append(BlobFile(filename, sep=sep))

    frame_start = np.min([f.frame_start for f in blob_files])
    frame_end = np.max([f.frame_end for f in blob_files])

    frame_dict = {frame: {} for frame in range(frame_start, frame_end)}  # Dict of files

    for i in range(frame_start, frame_end):
        for file in blob_files:
            if i in file.data[file.frame_header]:
                data = file.data.loc[file.data[file.frame_header] == i]
                points = np.array([data[file.x_header], data[file.y_header]]).T
                diameters = np.array(data[file.size_header])
                frame_dict[i][file.name] = BlobsFrame(points, diameters)
    return frame_dict


def write_blob_data(frame_dict: dict, filename: str, error_scale: float = 0) -> None:
    """
    Write a dict of BlobsFrames to a CSV file (predominantly used to simualate
    fake data.

    The CSV is saved with columns ["Frame", "x_cord:", "y_cord", "Size"]

    Parameters
    ----------
    blob_dict : dict with entries {int: BlobFrame}
        List of BlobFrames
    filename : str
        Filename to save folder
    """
    header = ["Frame", "x_cord", "y_cord", "Size"]
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_no, frame in frame_dict.items():
            for i in range(frame.n):
                writer.writerow(
                    [
                        frame_no,
                        frame.points[i, 0] + np.random.normal(0, error_scale),
                        frame.points[i, 1] + np.random.normal(0, error_scale),
                        frame.diameters[i],
                    ]
                )


class SpatialMatch:
    def __init__(self, B, Vs):
        self.B = B
        self.Vs = Vs
        self.n_V = len(Vs)

    def run_match(self, blob_data, p_est, Q_est, plots=[]):
        if len(blob_data) != self.n_V:
            print("ERROR: number of blob frames does not match viewpoints")
            raise ValueError
        p, Q = self.match_alg1(blob_data, p_est, Q_est, plots)
        plt.close()
        return p, Q

    def match_alg1(self, blob_data, p0, Q0, plots):
        # A simple traker which solves translation and rotation simultaneously
        # TODO: We NEED to develop better spatial match algorithms in the future

        # Extract the blob data from the  into 'unfiltered' lists
        X_ds_unfilt = []
        Y_ds_unfilt = []
        D_ds_unfilt = []
        for blobs in blob_data:
            X_d = blobs.points[:, 0]
            Y_d = blobs.points[:, 1]
            D_d = blobs.diameters
            X_ds_unfilt.append(X_d)
            Y_ds_unfilt.append(Y_d)
            D_ds_unfilt.append(D_d)

        # Filtering step - use a subselection of data blobs
        # TODO: This step could probably be improved for robustness
        # TODO: Improve notation with 2D position vectors
        X_ds = []
        Y_ds = []
        D_ds = []
        for i in range(self.n_V):
            view = self.Vs[i]
            X_d = X_ds_unfilt[i]
            Y_d = Y_ds_unfilt[i]
            D_d = D_ds_unfilt[i]

            p_i = np.array([X_d, Y_d])
            plots[i].update_observation(p_i)

            # Update model and views
            p = np.array(p0)
            self.B.update(p, Q0)
            p_p = view.get_blobs().points.T

            D = np.array(D_d)  # Diameters from image
            p_d = np.array([X_d, Y_d]).T  # Blob vectors from measured data

            # Find the distancer between each combination of blobs
            X_dif = np.subtract.outer(p_p[0], p_d.T[0])
            Y_dif = np.subtract.outer(p_p[1], p_d.T[1])
            r_dif = (X_dif**2 + Y_dif**2) ** 0.5

            # Find the closest image blob to each model blob
            # (Using each image blob only once)
            pairs = list(set([np.argmin(r) for r in r_dif]))

            # Only use the image blobs closest to the projected blobs
            X_ds.append([X_d[i] for i in pairs])
            Y_ds.append([Y_d[i] for i in pairs])
            D_ds.append([D_d[i] for i in pairs])

        def get_cost(C):
            Qs = Q0 * C[0:4]
            Qs = np.array(Qs) / np.linalg.norm(Qs)
            error = 0

            for i in range(self.n_V):
                view = self.Vs[i]
                X_d = X_ds[i]
                Y_d = Y_ds[i]
                D_d = D_ds[i]

                ps = p0 * C[4:7]
                self.B.update(ps, Qs)
                blobs = view.get_blobs()
                p_p = blobs.points

                blob_map = []
                min_norms = []
                for x, y, d in zip(X_d, Y_d, D_d):  # Iterate through image blobs
                    # TODO: It wold be better to construct a matrix of distances
                    # and start from the closest blob. Instead of starting with
                    # an arbitrary blob.
                    if len(p_p) > 0:
                        p_d = np.array([x, y])
                        delta_r = p_p - p_d
                        norms = np.linalg.norm(
                            delta_r, axis=1
                        )  # Distance image blob and projected blobs
                        blob_map.append(np.argmin(norms))
                        j = np.argmin(norms)
                        p_p = np.delete(
                            p_p.T, j, axis=1
                        ).T  # Option: only allow each dot to be used once
                        min_norms.append(np.min(norms))
                        # OPTIONAL: Weight by surface norm
                rms = np.mean(np.array(min_norms) ** 2) ** 0.5
                min_norms = np.array(min_norms)[np.where(np.abs(min_norms) < 2 * rms)]
                min_norms = min_norms**2
                error_ = np.mean(min_norms[0 : np.min((blobs.n, blobs.n))])
                error = error + error_
            return error

        def callback_plot(X):
            for view, X_d, Y_d, plot in zip(self.Vs, X_ds, Y_ds, plots):
                blobs_p = view.get_blobs()
                frame_p = view.get_mesh()
                cent = view.get_CoM()
                p_i = np.array([X_d, Y_d])
                plot.update_observation(p_i)
                plot.update_projection(blobs_p)
                plot.update_mesh(*frame_p)
                plot.update_CoM(cent)
                input("Press to continue")
                #  time.sleep(0.01)

        C0 = np.ones(7)
        sol = minimize(
            get_cost,
            C0,
            method="Powell",
            callback=callback_plot,
            options={"xtol": 1e-8, "ftol": 1e-6},
        )

        C = sol.x
        Q_final = Q0 * C[0:4]
        Q_final = np.array(Q_final) / np.linalg.norm(Q_final)
        p_final = p0 * C[4:7]
        return p_final, Q_final
