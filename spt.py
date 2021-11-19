# Adapted from Lars's code

import matplotlib

matplotlib.use('TkAgg')

from itertools import repeat
from matplotlib.path import Path
from matplotlib.widgets import Slider
import matplotlib.pylab as pl
from multiprocessing import Pool
import numpy as np
from pims import TiffStack, open
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import shutil
import trackpy as tp
import pandas as pd
import glob

"""
Tracking

"""


class SPT:
    """

    Tracking parameters:

    dist               search_range parameter for trackpy, distance a particle can move between frames in pixels
    featSize           diameter parameter for trackpy, approx. feature size
    maxsize            maxsize parameter for trackpy
    memory             memory parameter for trackpy
    minTrackLength     minimum track length for trackpy (func filter_stubs)
    threshold          minmass parameter for trackpy


    Movie parameters:

    direc              path of .tif stack file
    endFrame
    startFrame         first frame to take into consideration for analysis
    coors


    Computation:

    cores              number of parallel processes to be started if parallel is true
    parallel           boolean, to run tracking on more than one core



    """

    def __init__(self, direc=None, roi=None, threshold=600, feat_size=5, dist=5, separation=None, memory=3, mtl=50,
                 start_frame=0, end_frame=None, parallel=False, cores=None, save_path=None):

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
        self.ROI_invert = False

        # Computation
        self.cores = cores
        self.parallel = parallel

        # Saving
        self.save_path = save_path

        # Output data
        self.features = None
        self.features_all = None
        self.trajectories = pd.DataFrame([])

    def run(self):
        self.find_feats()
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

    def plot_calibration(self):
        PlotCalibration(self.frames, featSize=5, threshold=250)

    def plot_features(self):
        PlotFeatures(self.frames, self.trajectories)

    def plot_trajectories(self):
        tp.plot_traj(self.trajectories)

    def save(self):
        if self.save_path is not None:

            # Make results folder
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

            # Save tracking data
            self.trajectories.to_csv(self.save_path + '/trajectories.csv')


"""
Interactive

"""


def view_stack(stack, vmin=None, vmax=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    sframe = Slider(axframe, 'Time point', 0, len(stack), valinit=0, valfmt='%d')

    def update(i):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        ax.imshow(stack[int(i)], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    sframe.on_changed(update)
    plt.show()

    return int(sframe.val)


class ROI:
    """
    Instructions:
    - click to lay down points
    - backspace at any time to remove last point
    - press enter to select area (if spline=True will fit spline to points, otherwise will fit straight lines)
    - at this point can press backspace to go back to laying points
    - press enter again to close and return ROI

    :param stack: input image
    :param spline: if true, fits spline to inputted coordinates
    :return: cell boundary coordinates
    """

    def __init__(self, stack, spline, start_frame=0, end_frame=None, periodic=True):

        self.stack = stack
        self.spline = spline
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.periodic = periodic

        # Internal
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

        # Set up figure
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Calculate intensity ranges
        self.vmin, self.vmax = None, None

        # Stack
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        if self.end_frame is None:
            self.end_frame = len(self.stack) - 1
        self.sframe = Slider(self.axframe, 'Frame', self.start_frame, self.end_frame, valinit=self.start_frame,
                             valfmt='%d')
        self.sframe.on_changed(self.select_frame)
        self.select_frame(self.start_frame)

        # Show figure
        self.fig.canvas.set_window_title('Specify ROI')
        self.fig.canvas.mpl_connect('close_event', lambda event: self.fig.canvas.stop_event_loop())
        self.fig.canvas.start_event_loop(timeout=-1)

    def select_frame(self, i):
        self.ax.clear()
        self.ax.imshow(self.stack[int(i)], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.03, 0.97,
                     'Specify ROI clockwise (4 points minimum)'
                     '\nClick to lay points'
                     '\nBACKSPACE: undo'
                     '\nENTER: Save and continue',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8, va='top', ha='left')
        self.display_points()
        self.fig.canvas.draw()

    def button_press_callback(self, event):
        if not self._fitted:
            if isinstance(event.inaxes, type(self.ax)):
                # Add points to list
                self.xpoints.extend([event.xdata])
                self.ypoints.extend([event.ydata])

                # Display points
                self.display_points()
                self.fig.canvas.draw()

    def key_press_callback(self, event):
        if event.key == 'backspace':
            if not self._fitted:

                # Remove last drawn point
                if len(self.xpoints) != 0:
                    self.xpoints = self.xpoints[:-1]
                    self.ypoints = self.ypoints[:-1]
                self.display_points()
                self.fig.canvas.draw()
            else:

                # Remove line
                self._fitted = False
                self._line.pop(0).remove()
                self.roi = None
                self.fig.canvas.draw()

        if event.key == 'enter':
            if not self._fitted:
                roi = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    self.roi = spline_roi(roi, periodic=self.periodic)

                # Display line
                self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                self.fig.canvas.draw()

                self._fitted = True

                # print(self.roi)

                plt.close(self.fig)  # comment this out to see spline fit
            else:
                # Close figure window
                plt.close(self.fig)

    def display_points(self):

        # Remove existing points
        try:
            self._point0.remove()
            self._points.remove()
        except (ValueError, AttributeError) as error:
            pass

        # Plot all points
        if len(self.xpoints) != 0:
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='lime', s=10)
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r', s=10)


def def_roi(img, spline=True, start_frame=0, end_frame=None, periodic=True):
    r = ROI(img, spline=spline, start_frame=start_frame, end_frame=end_frame, periodic=periodic)
    return r.roi


class PlotCalibration:
    """


    """

    def __init__(self, frames, threshold=250, featSize=5, maxSize=None, separation=None):

        # Data
        self.frames = frames
        self.threshold = threshold
        self.featSize = featSize
        self.maxSize = maxSize
        if separation is not None:
            self.separation = separation
        else:
            self.separation = featSize + 1

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Frame slider
        self.axframe = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')

        # Threshold slider
        self.axframe2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sframe2 = Slider(self.axframe2, 'Threshold', 0, 1000, valinit=self.threshold, valfmt='%d')

        # Featsize slider
        self.axframe3 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe3 = Slider(self.axframe3, 'Feat Size', 0, 10, valinit=self.featSize, valfmt='%d')

        # Separation slider
        self.axframe4 = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.sframe4 = Slider(self.axframe4, 'Separation', 0, 10, valinit=self.separation, valfmt='%d')

        # Plot parameters
        self.display_points = True
        self.current_frame = 0

        # Display
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.sframe.on_changed(self.update_current_frame)
        self.sframe2.on_changed(self.update_threshold)
        self.sframe3.on_changed(self.update_featsize)
        self.sframe4.on_changed(self.update_separation)
        self.refresh()
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.refresh()
            self.fig.canvas.draw()

        if event.key == ',':
            self.current_frame -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_current_frame)
            self.refresh()
            self.fig.canvas.draw()

        if event.key == '.':
            self.current_frame += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_current_frame)
            self.refresh()
            self.fig.canvas.draw()

    def update_current_frame(self, i):
        self.current_frame = int(i)
        self.refresh()

    def update_threshold(self, i):
        self.threshold = i
        self.refresh()

    def update_featsize(self, i):
        self.featSize = int(i) // 2 * 2 + 1
        self.refresh()

    def update_separation(self, i):
        self.separation = i
        self.refresh()

    def refresh(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.frames[self.current_frame], cmap='gray')

        if self.display_points:
            f = tp.locate(self.frames[self.current_frame], diameter=self.featSize, invert=False, minmass=self.threshold,
                          maxsize=self.maxSize, separation=self.separation)
            self.ax.scatter(f['x'], f['y'], facecolors='none', edgecolor='r', s=100, linewidths=1)

            print(np.median(np.array(tp.proximity(f))))

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


class PlotFeatures:
    """
    Plot tracked particles

    """

    def __init__(self, frames, trajectories):

        # Data
        self.frames = frames
        self.trajectories = trajectories

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Plot parameters
        self.display_points = True
        self.display_traj = True
        self.i = 0

        # Display
        self.sframe.on_changed(self.update)
        self.update(0)
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == 't':
            self.display_traj = not self.display_traj
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == ',':
            self.i -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.i, valfmt='%d')
            self.sframe.on_changed(self.update)
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == '.':
            self.i += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.i, valfmt='%d')
            self.sframe.on_changed(self.update)
            self.update(self.i)
            self.fig.canvas.draw()

    def update(self, i):
        self.i = i
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        f = self.trajectories[self.trajectories.frame == int(i)]
        self.ax.imshow(self.frames[int(i)], cmap='gray')

        if self.display_traj:
            cmap = pl.cm.hsv(np.linspace(0, 1, 20))
            for p in f['particle']:
                single_traj = self.trajectories[self.trajectories['particle'] == p]
                single_traj = single_traj[single_traj['frame'] <= self.i]
                self.ax.plot(single_traj['x'], single_traj['y'], c=cmap[(p % 20)], linewidth=0.5)

        if self.display_points:
            a = self.ax.scatter(f['x'], f['y'], c=(f['particle'] % 20), cmap='hsv', vmin=0, vmax=20, s=100,
                                linewidths=1)
            a.set_facecolor('none')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


class PlotFeatures2colour:
    """
    Plot tracked particles

    """

    def __init__(self, frames, trajectories1, trajectories2):

        # Data
        self.frames = frames
        self.trajectories1 = trajectories1
        self.trajectories2 = trajectories2

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)

        # Plot parameters
        self.display_points = True
        self.display_traj = True
        self.i = 0

        # Display
        self.sframe.on_changed(self.update)
        self.update(0)
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == 't':
            self.display_traj = not self.display_traj
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == ',':
            self.i -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.i, valfmt='%d')
            self.sframe.on_changed(self.update)
            self.update(self.i)
            self.fig.canvas.draw()

        if event.key == '.':
            self.i += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.i, valfmt='%d')
            self.sframe.on_changed(self.update)
            self.update(self.i)
            self.fig.canvas.draw()

    def update(self, i):
        self.i = i
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        f1 = self.trajectories1[self.trajectories1.frame == int(i)]
        f2 = self.trajectories2[self.trajectories2.frame == int(i)]
        self.ax.imshow(self.frames[int(i)], cmap='gray')

        if self.display_traj:
            for p in f1['particle']:
                single_traj = self.trajectories1[self.trajectories1['particle'] == p]
                single_traj = single_traj[single_traj['frame'] <= self.i]
                self.ax.plot(single_traj['x'], single_traj['y'], c='b', linewidth=0.5)
            for p in f2['particle']:
                single_traj = self.trajectories2[self.trajectories2['particle'] == p]
                single_traj = single_traj[single_traj['frame'] <= self.i]
                self.ax.plot(single_traj['x'], single_traj['y'], c='r', linewidth=0.5)

        if self.display_points:
            a = self.ax.scatter(f1['x'], f1['y'], c='none', edgecolors='b', s=100, linewidths=1)
            # a.set_facecolor('none')
            b = self.ax.scatter(f2['x'], f2['y'], c='none', edgecolors='r', s=100, linewidths=1)
            # b.set_facecolor('none')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


class PlotSimImg:
    """


    """

    def __init__(self, frames, threshold=250, featSize=5, maxSize=None, separation=None):

        # Data
        self.frames = frames
        self.threshold = threshold
        self.featSize = featSize
        self.maxSize = maxSize
        if separation is not None:
            self.separation = separation
        else:
            self.separation = featSize + 1

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Frame slider
        self.axframe = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')

        # Threshold slider
        self.axframe2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sframe2 = Slider(self.axframe2, 'Threshold', 0, 1000, valinit=self.threshold, valfmt='%d')

        # Featsize slider
        self.axframe3 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe3 = Slider(self.axframe3, 'Feat Size', 0, 10, valinit=self.featSize, valfmt='%d')

        # Separation slider
        self.axframe4 = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.sframe4 = Slider(self.axframe4, 'Separation', 0, 10, valinit=self.separation, valfmt='%d')

        # Plot parameters
        self.display_points = True
        self.current_frame = 0

        # Display
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.sframe.on_changed(self.update_current_frame)
        self.sframe2.on_changed(self.update_threshold)
        self.sframe3.on_changed(self.update_featsize)
        self.sframe4.on_changed(self.update_separation)
        self.refresh()
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.refresh()
            self.fig.canvas.draw()

        if event.key == ',':
            self.current_frame -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_current_frame)
            self.refresh()
            self.fig.canvas.draw()

        if event.key == '.':
            self.current_frame += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_current_frame)
            self.refresh()
            self.fig.canvas.draw()

    def update_current_frame(self, i):
        self.current_frame = int(i)
        self.refresh()

    def update_threshold(self, i):
        self.threshold = i
        self.refresh()

    def update_featsize(self, i):
        self.featSize = int(i) // 2 * 2 + 1
        self.refresh()

    def update_separation(self, i):
        self.separation = i
        self.refresh()

    def refresh(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()

        f = tp.locate(self.frames[self.current_frame], diameter=self.featSize, invert=False, minmass=self.threshold,
                      maxsize=self.maxSize, separation=self.separation)
        self.ax.imshow(image_from_speckles(f, [512, 512]).T, cmap='gray')

        if self.display_points:
            self.ax.scatter(f['x'], f['y'], facecolors='none', edgecolor='r', s=100, linewidths=1)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


class AdjustMTL:
    """


    """

    def __init__(self, frames, trajectories):

        # Data
        self.frames = frames
        self.trajectories = trajectories
        self.traj_filtered = trajectories
        self.min_tl = 0
        self.max_tl = 100

        # Initial filter
        self.traj_filtered = self.trajectories.groupby('particle').filter(
            lambda x: self.min_tl <= x.frame.count() <= self.max_tl)

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Frame slider
        self.axframe = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')

        # Min track length slider
        self.axframe2 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe2 = Slider(self.axframe2, 'Min track length', 0, 100, valinit=self.min_tl, valfmt='%d')

        # Max track length slider
        self.axframe3 = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.sframe3 = Slider(self.axframe3, 'Max track length', 0, 100, valinit=self.max_tl, valfmt='%d')

        # Plot parameters
        self.display_points = False
        self.display_traj = False
        self.current_frame = 0

        # Display
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.sframe.on_changed(self.update_frame)
        self.sframe2.on_changed(self.update_min_tl)
        self.sframe3.on_changed(self.update_max_tl)
        self.refresh()
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.refresh()
            self.fig.canvas.draw()

        if event.key == 't':
            self.display_traj = not self.display_traj
            self.refresh()
            self.fig.canvas.draw()

        if event.key == ',':
            self.current_frame -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_frame)
            self.refresh()
            self.fig.canvas.draw()

        if event.key == '.':
            self.current_frame += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(self.update_frame)
            self.refresh()
            self.fig.canvas.draw()

    def update_min_tl(self, i):
        self.min_tl = int(i)
        self.traj_filtered = self.trajectories.groupby('particle').filter(
            lambda x: self.min_tl <= x.frame.count() <= self.max_tl)
        self.refresh()

    def update_max_tl(self, i):
        self.max_tl = int(i)
        self.traj_filtered = self.trajectories.groupby('particle').filter(
            lambda x: self.min_tl <= x.frame.count() <= self.max_tl)
        self.refresh()

    def update_frame(self, frame):
        self.current_frame = int(frame)
        self.refresh()

    def refresh(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()

        f = self.traj_filtered[self.traj_filtered.frame == int(self.current_frame)]
        self.ax.imshow(self.frames[int(self.current_frame)], cmap='gray')

        if self.display_traj:
            cmap = pl.cm.hsv(np.linspace(0, 1, 20))
            for p in f['particle']:
                single_traj = self.traj_filtered[self.traj_filtered['particle'] == p]
                single_traj = single_traj[single_traj['frame'] <= self.current_frame]
                self.ax.plot(single_traj['x'], single_traj['y'], c=cmap[(p % 20)], linewidth=0.5)

        if self.display_points:
            a = self.ax.scatter(f['x'], f['y'], c=(f['particle'] % 20), cmap='hsv', vmin=0, vmax=20, s=100,
                                linewidths=1)
            a.set_facecolor('none')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


def plot_tracks(traj, c=None):
    unstacked = traj.set_index(['particle', 'frame'])[['x', 'y']].unstack()
    for i, trajectory in unstacked.iterrows():
        if c is not None:
            plt.plot(trajectory['x'], trajectory['y'], c=c)
        else:
            plt.plot(trajectory['x'], trajectory['y'], c=c)


def image_from_speckles(speckles, size):
    pos = np.c_[speckles['x'], speckles['y']]

    img = sum_gaussians(size, 1, pos, 5)
    return img


def sum_gaussians(gridsize, resolution, partPos, Sigma):
    '''
    Author: Lars

    Creates image convolved with an artificial point spread function.
    PSF has breadth Sigma, within each pixel of image gridsize
    '''
    import bisect
    import scipy.stats as stats

    # Initialize arrays
    sz = np.shape(partPos)
    numSteps = int(4 * Sigma / resolution + 1)  # 1-D interpolation length
    interpLen = int(numSteps ** 2)  # len of interpolation around point
    arraysize = int(interpLen * sz[0])  # len of containers for x, y, p
    xall = np.zeros(arraysize)  # container for x
    yall = np.zeros(arraysize)  # container for y
    pall = np.zeros(arraysize)  # container for probabilities
    imMat = np.zeros(gridsize)  # image matrix to project particles on

    # For each particle, create a meshgrid of size 2*Sigma around that
    # particle, link them all together and evaluate gaussian of
    # spread Sigma and mean partPos at each point of the meshgrid.
    for i in range(sz[0]):
        X = np.linspace(partPos[i, 0] - 2 * Sigma, partPos[i, 0] + 2 * Sigma, numSteps)
        Y = np.linspace(partPos[i, 1] - 2 * Sigma, partPos[i, 1] + 2 * Sigma, numSteps)
        x, y = np.meshgrid(X, Y)
        xall[i * interpLen:(i + 1) * interpLen] = x.flatten(order='F')
        yall[i * interpLen:(i + 1) * interpLen] = y.flatten(order='F')
        pall[i * interpLen:(i + 1) * interpLen] = stats.multivariate_normal.pdf(
            np.c_[x.flatten('F'), y.flatten('F')],
            partPos[i, :], Sigma)
    # Sort by ascending x-values
    xSortInds = np.argsort(xall, kind='mergesort')
    xsorted = xall[xSortInds]
    ysorted = yall[xSortInds]
    psorted = pall[xSortInds]

    # Project particles onto image.
    # Iterate over x dimension of image
    for i in range(gridsize[0]):
        temp_min = bisect.bisect_left(xsorted, i) - 1
        temp_max = bisect.bisect_right(xsorted, i + 1)
        for j in range(gridsize[1]):
            if not (temp_min == -1 or temp_max == -1):
                ytemp = ysorted[temp_min:temp_max + 1]
                ptemp = psorted[temp_min:temp_max + 1]
                # Sum all values from pall that have an x and y value within
                # the current pixel (i, j).
                imMat[i, j] = np.sum(ptemp[np.logical_and(ytemp > j,
                                                          ytemp <= j + 1)])
    return imMat


"""
Misc functions

"""


def interp_roi(roi, periodic=True):
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
    Linear interpolation

    Ability to specify number of points

    :param roi:
    :return:
    """

    if periodic:
        c = np.append(roi, [roi[0, :]], axis=0)
    else:
        c = roi

    # Calculate distance between points in pixel units
    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances_cumsum = np.append([0], np.cumsum(distances))
    px = sum(distances) / round(sum(distances))  # effective pixel size
    newpoints = np.zeros((int(round(sum(distances))), 2))
    newcoors_distances_cumsum = 0

    for d in range(int(round(sum(distances)))):
        index = sum(distances_cumsum - newcoors_distances_cumsum <= 0) - 1
        newpoints[d, :] = (
                roi[index, :] + ((newcoors_distances_cumsum - distances_cumsum[index]) / distances[index]) * (
                c[index + 1, :] - c[index, :]))
        newcoors_distances_cumsum += px

    return newpoints


def spline_roi(roi, periodic=True):
    """
    Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

    :param roi:
    :return:
    """

    # Append the starting x,y coordinates
    if periodic:
        x = np.r_[roi[:, 0], roi[0, 0]]
        y = np.r_[roi[:, 1], roi[0, 1]]
    else:
        x = roi[:, 0]
        y = roi[:, 1]

    # Fit spline
    tck, u = splprep([x, y], s=0, per=periodic)

    # Evaluate spline
    xi, yi = splev(np.linspace(0, 1, 1000), tck)

    # Interpolate
    return interp_roi(np.vstack((xi, yi)).T, periodic=periodic)


def polar_roi(roi, pole):
    """
    Creates ROI for anterior or posterior pole
    Assumes first point specified is the posterior

    """

    # # PCA to find long axis
    # M = (roi - np.mean(roi.T, axis=1)).T
    # [latent, coeff] = np.linalg.eig(np.cov(M))
    # score = np.dot(coeff.T, M)
    #
    # # Find most extreme points
    # a = np.argmin(np.minimum(score[0, :], score[1, :]))
    # b = np.argmax(np.maximum(score[0, :], score[1, :]))
    #
    # # Find the one closest to user defined posterior
    # dista = np.hypot((roi[0, 0] - roi[a, 0]), (roi[0, 1] - roi[a, 1]))
    # distb = np.hypot((roi[0, 0] - roi[b, 0]), (roi[0, 1] - roi[b, 1]))
    #
    # # Rotate coordinates
    # if dista < distb:
    #     newcoors = np.roll(roi, len(roi[:, 0]) - a, 0)
    # else:
    #     newcoors = np.roll(roi, len(roi[:, 0]) - b, 0)

    # Polar ROI
    npoints = roi.shape[0]
    if pole == 'a':
        return roi[int(npoints / 4):-int(npoints / 4), :]
    elif pole == 'p':
        return np.r_[roi[:int(npoints / 4), :], roi[-int(npoints / 4):, :]]


def median_proximity(frames, ROI, time_gap, dist, minmass, separation):
    """
    Could do this much more efficiently without a for loop

    """

    bbPath = Path(np.asarray(list(zip(*(ROI[:, 0], ROI[:, 1])))))

    times = np.zeros(len(frames[::time_gap]))
    proximities = np.zeros(len(frames[::time_gap]))

    for i, f in enumerate(frames[::time_gap]):
        try:
            # Find speckles
            speckles = tp.locate(f, dist, invert=False, minmass=minmass, separation=separation)

            # Apply ROI
            x_y_tuples = list(zip(*(speckles['x'].values, speckles['y'].values)))
            mask = [bbPath.contains_point(np.asarray(i)) for i in x_y_tuples]
            speckles = speckles[mask]

            # Proximity
            times[i] = i * time_gap
            proximities[i] = np.median(tp.proximity(speckles)['proximity'].values)
        except:
            times[i] = i * time_gap
            proximities[i] = np.nan

    return times, proximities


def direcslist(parent, levels=0, exclude=('!',), include=None):
    """
    Gives a list of directories in a given directory (full path), filtered according to criteria

    :param parent: parent directory to search
    :param levels: goes down this many levels
    :param exclude: exclude directories containing this string
    :param include: exclude directories that don't contain this string
    :return:
    """
    lis = glob.glob('%s/*/' % parent)

    for level in range(levels):
        newlis = []
        for e in lis:
            newlis.extend(glob.glob('%s/*/' % e))
        lis = newlis
        lis = [x[:-1] for x in lis]

    # Filter
    if exclude is not None:
        for i in exclude:
            lis = [x for x in lis if i not in x]

    if include is not None:
        for i in include:
            lis = [x for x in lis if i in x]

    return sorted(lis)


def bar(ax, data, pos, c, size=5, label=None):
    ax.bar(pos, np.mean(data), width=3, color=c, alpha=0.1, label=label)
    ax.scatter(pos - 1 + 2 * np.random.rand(len(data)), data, facecolors='none', edgecolors=c, linewidth=1, zorder=2,
               s=size)

    ax.set_xticks(list(ax.get_xticks()) + [pos])
    ax.set_xlim([0, max(ax.get_xticks()) + 4])
