# Adapted from Lars's code

from itertools import repeat
from matplotlib.path import Path
from multiprocessing import Pool
import numpy as np
from pims import TiffStack
import os
import trackpy as tp
import pandas as pd
from spt.roi import def_roi
from spt.funcs import polar_roi
from spt.interactive import plot_calibration, plot_features
import multiprocessing
import matplotlib.pyplot as plt

"""
Tracking

"""


class SPT:
    """

    Movie parameters:

    direc              path of .tif stack file
    startFrame         first frame to take into consideration for analysis
    endFrame           last frame to take into consideration for analysis
    roi                coordinates defining the roi. Output from def_roi() function, which can be saved as a .txt file
                       and reimported


    Tracking parameters:

    dist               search_range parameter for trackpy, distance a particle can move between frames in pixels
    featSize           diameter parameter for trackpy, approx. feature size
    memory             memory parameter for trackpy
    mtl                minimum track length for trackpy (func filter_stubs)
    threshold          minmass parameter for trackpy
    separation         separation parameter for trackpy


    Computation:

    cores              number of parallel processes to be started if parallel is true
    parallel           boolean, to run tracking on more than one core


    Results:

    save_path          path to save the results (will create a file called trajectories.csv)


    """

    def __init__(self, direc=None, roi=None, threshold=600, feat_size=5, dist=5, separation=None, memory=3, mtl=50,
                 start_frame=0, end_frame=None, parallel=False, cores=None, save_path=None, invert_roi=False):

        # Import data
        self.direc = direc
        self.end_frame = end_frame
        self.start_frame = start_frame
        if self.direc:
            self.frames = TiffStack(self.direc)
            self.frames = self.frames[self.start_frame:self.end_frame]

        # Particle localisation parameters
        self.threshold = threshold
        self.feat_size = feat_size
        self.maxsize = None
        self.separation = separation

        # Tracking parameters
        self.dist = dist
        self.memory = memory
        self.mtl = mtl
        self.adaptive_stop = None
        self.link_strategy = 'auto'

        # ROI
        self.ROI = roi
        self.ROI_original = roi
        self.ROI_invert = invert_roi

        # Computation
        if cores is not None:
            self.cores = cores
        else:
            self.cores = multiprocessing.cpu_count()
        self.parallel = parallel

        # Saving
        self.save_path = save_path

        # Output data
        self.features = None
        self.features_all = None
        self.trajectories = pd.DataFrame([])

    def run(self):
        self.find_feats()
        if self.ROI is not None:
            self.apply_roi()
        self.link_feats()
        self.save()

    def apply_roi(self):
        """
        Filter all found features by whether they have been found
        within self.ROI

        """

        bbPath = Path(np.asarray(
            list(zip(*(self.ROI[:, 0], self.ROI[:, 1])))))
        x_y_tuples = list(zip(*(self.features_all['x'].values,
                                self.features_all['y'].values)))
        mask = [bbPath.contains_point(np.asarray(i)) for i in x_y_tuples]
        if self.ROI_invert:
            self.features = self.features_all[np.invert(mask)]
        else:
            self.features = self.features_all[mask]

    def def_roi(self, n=0):
        """
        Define a ROI in the nth frame

        """
        self.ROI = def_roi(self.frames[n])
        self.ROI_original = self.ROI

    def polar_roi(self, pole):
        """
        Creates ROI for anterior or posterior pole

        """
        self.ROI = polar_roi(self.ROI_original, pole)

    def find_feats(self):
        """

        """

        if self.parallel:
            # Create list of frames to be analysed by separate processes
            f_list = []
            # Get size of chunks
            s = np.ceil(len(self.frames) / self.cores)
            for ii in range(0, self.cores - 1):
                # Issue with index, check output!
                f_list.append(self.frames[int(s * ii):int(s * (ii + 1))])
            # Last worker gets slightly more frames
            f_list.append(self.frames[int(s * (self.cores - 1)):])
            # Create pool, use starmap to pass more than one parameter, do work
            pool = Pool(processes=self.cores)
            res = pool.starmap(tp.batch, zip(f_list,
                                             repeat(self.feat_size),
                                             repeat(self.threshold),
                                             repeat(self.maxsize),
                                             repeat(self.separation)))
            # Concatenate results and assign to features
            self.features_all = pd.concat(res)
            # Close and join pool
            pool.close()
            pool.join()
        else:
            self.features_all = tp.batch(self.frames[:], self.feat_size, minmass=self.threshold, maxsize=self.maxsize,
                                         separation=self.separation, invert=False)

        self.features_all['frame'] -= self.start_frame
        self.features = self.features_all

    def link_feats(self):
        """
        Link individual frames to build trajectories, filter out stubs shorter than minTrackLength.

        """

        t = tp.link_df(self.features, search_range=self.dist, memory=self.memory, link_strategy=self.link_strategy,
                       adaptive_stop=self.adaptive_stop)
        trajectories = tp.filter_stubs(t, self.mtl)
        self.trajectories = trajectories

    def plot_calibration(self, jupyter=False):
        plot_calibration(self.frames, featSize=self.feat_size, threshold=self.threshold, jupyter=jupyter)

    def plot_features(self, jupyter):
        plot_features(self.frames, self.trajectories, jupyter=jupyter)

    def plot_trajectories(self):
        fig, ax = plt.subplots()
        tp.plot_traj(self.trajectories, ax=ax)

    def save(self):
        if self.save_path is not None:

            # Make results folder
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

            # Save tracking data
            self.trajectories.to_csv(self.save_path + '/trajectories.csv')
