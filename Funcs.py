# Adapted from Lars's code

import matplotlib

matplotlib.use('TkAgg')

from itertools import repeat
from matplotlib.path import Path
from matplotlib.widgets import Slider
import matplotlib.pylab as pl
from multiprocessing import Pool
import numpy as np
from pandas import concat, DataFrame
from pims import TiffStack
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
import scipy
import os
import shutil

import seaborn as sns
import trackpy as tp

"""
TO DO:

For each movie, make text files called Notes
def_ROI should have slider



Default analysis (Halo):

threshold=400
featSize=5
minTrackLength=30
memory=3
dist=5

"""

"""
Base class

"""


class ParticleFinder(object):
    """

    Tracking parameters:

    adaptive_stop
    dist               search_range parameter for trackpy, distance a particle can move between frames in pixels
    featSize           diameter parameter for trackpy, approx. feature size
    link_strat
    maxsize            maxsize parameter for trackpy
    memory             memory parameter for trackpy
    minTrackLength     minimum track length for trackpy (func filter_stubs)
    threshold          minmass parameter for trackpy
    numFrames


    Movie parameters:

    direc              path of .tif stack file
    timestep           real time difference between frames (seconds)
    endFrame
    pixelSize          pixel size of microscope in microns
    startFrame         first frame to take into consideration for analysis
    coors


    Computation:

    cores              number of parallel processes to be started if parallel is true
    parallel           boolean, to run feature finding on more than one core



    """

    def __init__(self, direc=None, coors=None, threshold=600, adaptive_stop=None, dist=5, endFrame=None, featSize=5,
                 link_strat='auto', maxsize=None, memory=3, minTrackLength=50, cores=8,
                 parallel=False, pixelSize=0.124, startFrame=0, timestep=None, numFrames=10):

        # Tracking parameters
        self.adaptive_stop = adaptive_stop
        self.dist = dist
        self.featSize = featSize
        self.link_strat = link_strat
        self.maxsize = maxsize
        self.memory = memory
        self.minTrackLength = minTrackLength
        self.threshold = threshold
        self.numFrames = numFrames

        # Movie parameters
        self.direc = direc
        self.timestep = timestep
        self.endFrame = endFrame
        self.pixelSize = pixelSize
        self.startFrame = startFrame
        self.ROI = coors
        self.ROI_original = coors
        self.ROIinvert = False

        # Computation
        self.cores = cores
        self.parallel = parallel

        # Import data
        if self.direc:
            self.frames = TiffStack(self.direc)
            self.frames = self.frames[self.startFrame:self.endFrame]

        # Output data
        self.features = None
        self.features_all = None
        self.trajectories = DataFrame([])
        self.im = DataFrame([])
        self.D = None
        self.alpha = None
        self.resids = None

    def apply_ROI(self):
        """
        Filter all found features by whether they have been found
        within this self.ROI

        """

        bbPath = Path(np.asarray(
            list(zip(*(self.ROI[:, 0], self.ROI[:, 1])))))
        x_y_tuples = list(zip(*(self.features_all['x'].values,
                                self.features_all['y'].values)))
        mask = [bbPath.contains_point(np.asarray(i)) for i in x_y_tuples]
        if self.ROIinvert:
            self.features = self.features_all[np.invert(mask)]
        else:
            self.features = self.features_all[mask]

    def def_ROI(self, n=0):
        """
        Define a ROI in the nth frame

        """
        self.ROI = def_ROI(self.frames[n])
        self.ROI_original = self.ROI

    def polar_ROI(self, pole):
        """
        Creates ROI for anterior or posterior pole

        """

        # PCA to find long axis
        M = (self.ROI_original - np.mean(self.ROI_original.T, axis=1)).T
        [latent, coeff] = np.linalg.eig(np.cov(M))
        score = np.dot(coeff.T, M)

        # Find most extreme points
        a = np.argmin(np.minimum(score[0, :], score[1, :]))
        b = np.argmax(np.maximum(score[0, :], score[1, :]))

        # Find the one closest to user defined posterior
        dista = np.hypot((self.ROI_original[0, 0] - self.ROI_original[a, 0]),
                         (self.ROI_original[0, 1] - self.ROI_original[a, 1]))
        distb = np.hypot((self.ROI_original[0, 0] - self.ROI_original[b, 0]),
                         (self.ROI_original[0, 1] - self.ROI_original[b, 1]))

        # Rotate coordinates
        if dista < distb:
            newcoors = np.roll(self.ROI_original, len(self.ROI_original[:, 0]) - a, 0)
        else:
            newcoors = np.roll(self.ROI_original, len(self.ROI_original[:, 0]) - b, 0)

        # Polar ROI
        npoints = self.ROI_original.shape[0]
        if pole == 'a':
            self.ROI = newcoors[int(npoints / 4):-int(npoints / 4), :]
        elif pole == 'p':
            self.ROI = np.r_[newcoors[:int(npoints / 4), :], newcoors[-int(npoints / 4):, :]]

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
                                             repeat(self.featSize),
                                             repeat(self.threshold),
                                             repeat(self.maxsize)))
            # Concatenate results and assign to features
            self.features_all = concat(res)
            # Close and join pool
            pool.close()
            pool.join()
        else:
            self.features_all = tp.batch(self.frames[:], self.featSize,
                                         minmass=self.threshold,
                                         maxsize=self.maxsize,
                                         invert=False)

        self.features_all['frame'] -= self.startFrame
        self.features = self.features_all

    def plot_calibration(self):
        PlotCalibration(self.frames, featSize=5, threshold=250)

    def plot_features(self):
        PlotFeatures(self.frames, self.trajectories)

    def plot_trajectories(self):
        tp.plot_traj(self.trajectories)


"""
Diffusion analysis

"""


class DiffusionFitter(ParticleFinder):
    """


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analyze(self):
        """


        """

        self.find_feats()
        if self.ROI is not None:
            self.apply_ROI()
        self.link_feats()
        self.analyze_tracks()

    def link_feats(self):
        """
        Link individual frames to build trajectories, filter out stubs shorter
        than minTrackLength. Get Mean Square Displacement (msd).

        """

        t = tp.link_df(self.features, self.dist, memory=self.memory,
                       link_strategy=self.link_strat,
                       adaptive_stop=self.adaptive_stop)
        trajectories = tp.filter_stubs(t, self.minTrackLength)
        im = tp.imsd(trajectories, mpp=self.pixelSize, fps=1 / self.timestep, max_lagtime=100)
        self.trajectories = trajectories
        self.im = im

    def analyze_tracks(self):
        """
        Get diffusion coefficients

        """
        num_particles = self.trajectories['particle'].nunique()
        im_array = self.im.as_matrix()
        time = np.linspace(self.timestep, self.timestep * self.numFrames, self.numFrames)
        DA = np.zeros([num_particles, 2])
        self.resids = np.zeros([num_particles, 1])
        for j in range(0, num_particles):
            MSD = im_array[0:self.numFrames, j]
            results = np.polyfit(np.log10(time), np.log10(MSD), 1, full=True)  # w=np.sqrt(MSD)
            DA[j,] = [results[0][0], results[0][1]]
            self.resids[j] = results[1][0]

            # plt.plot(np.log10(time), np.log10(MSD))
            # plt.plot(np.log10(time), results[0][0] * np.log10(time) + results[0][1])
            # plt.show()
            #
            # plt.plot(time, MSD)
            # plt.plot(time, (10 ** results[0][1]) * (time ** results[0][0]))
            # plt.show()
        self.D = 10 ** DA[:, 1] / 4
        self.alpha = DA[:, 0]

    def analyze_tracks2(self):
        """
        Different fitting method

        """

        num_particles = self.trajectories['particle'].nunique()
        im_array = self.im.as_matrix()
        time = np.linspace(self.timestep, self.timestep * self.numFrames, self.numFrames)
        DA = np.zeros([num_particles, 2])
        for j in range(0, num_particles):
            MSD = im_array[0:self.numFrames, j]
            popt, pcov = curve_fit(lambda t, d, a: 4 * d * t ** a, time, MSD, p0=(0.1, 1))
            DA[j,] = [popt[0], popt[1]]

            # plt.plot(time, MSD)
            # plt.plot(time, 4 * popt[0] * time ** popt[1])
            # plt.show()

        self.D = DA[:, 0]
        self.alpha = DA[:, 1]

    def plot_msd(self):
        fig, ax = plt.subplots()
        fig.suptitle('MSD vs lag time', fontsize=20)
        ax.plot(self.im.index, self.im, 'k-', alpha=0.4)  # already in sec
        ax.set(ylabel='$\Delta$ $r^2$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()

    def save(self, direc):
        # Make results folder
        if os.path.isdir(direc):
            shutil.rmtree(direc)
        os.mkdir(direc)

        # Save results
        self.trajectories.to_csv(direc + '/trajectories.csv')
        self.im.to_csv(direc + '/im.csv')
        np.savetxt(direc + '/D.txt', self.D)
        np.savetxt(direc + '/alpha.txt', self.alpha)
        np.savetxt(direc + '/resids.txt', self.resids)


"""
Off rate analysis

"""


class OffRateFitter(ParticleFinder):
    """
    Extends ParticleFinder to implement Off rate calculation

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partCount = None
        self.fitTimes = None

        # self.koff = None
        # self.kph = None
        # self.nss = None

    def analyze(self):
        self.find_feats()
        self.partCount, _ = np.histogram(self.features.frame, bins=self.features.frame.max() + 1)
        self.fitTimes = np.arange(0, len(self.partCount) * self.timestep, self.timestep)

    def fit_off_rate(self, variant=1):
        """
        Fit differential equation to data by solving with odeint and
        using fmin to parameters that best fit time/intensity data or
        by using optimize.curve_fit. Different variants have use
        different free parameters.
        """

        '''
        Variant 1: fits differential equation to data, free parameters
        kOff, Nss, kPh, assumes infinite cytoplasmic pool
        '''
        if variant == 1:
            def dy_dt(y, t, kOff, Nss, kPh):
                # Calculates derivative for known y and params
                return kOff * Nss - (kOff + kPh) * y

            def objFunc(params, fitTimes, fitData):
                # Returns distance between solution of diff. equ. with
                # parameters params and the data fitData at times fitTimes
                # Do integration of dy_dt using parameters params
                y = integrate.odeint(dy_dt, params[1], fitTimes, args=(params[0], params[1], params[2]))
                # Get y-values at the times needed to compare with data
                return sum((y.flatten() - fitData) ** 2)

            # Set reasonable starting values for optimization
            kOffStart, NssStart, kPhStart = 0.01, self.partCount[0], 0.01
            # Optimize objFunc to find optimal kOffStart, NssStart, kPhStart
            x = scipy.optimize.fmin(objFunc, [kOffStart, NssStart, kPhStart],
                                    args=(self.fitTimes, self.partCount),
                                    disp=False)

            self.kOffVar1, self.NssVar1, self.kPhVar1 = (x[0], x[1], x[2])
            # Get solution using final parameter set determined by fmin
            self.fitSolVar1 = integrate.odeint(dy_dt, self.NssVar1, self.fitTimes,
                                               args=(self.kOffVar1, self.NssVar1, self.kPhVar1))

        '''
        Variant 2: fits solution of DE to data, fitting N(0), N(Inf) and koff,
        therefore being equivalent to variant=1
        '''
        if variant == 2:
            def exact_solution(times, koff, count0, countInf):
                return (count0 - countInf) * np.exp(-koff * count0 / countInf * times) + countInf

            popt, pcov = curve_fit(exact_solution, self.fitTimes, self.partCount)
            self.kOffVar2 = popt[0]
            self.fitSolVar2 = exact_solution(self.fitTimes, popt[0], popt[1], popt[2])

        '''
        Variant 3: fits solution of DE to data, assuming Nss=N(0) and
        Nss_bleach=N(end), only one free parameter: koff
        '''
        if variant == 3:
            def exact_solution(count0, countInf):
                def curried_exact_solution(times, koff):
                    return ((count0 - countInf) *
                            np.exp(-koff * count0 / countInf * times) + countInf)

                return curried_exact_solution

            popt, pcov = curve_fit(exact_solution(self.partCount[0], self.partCount[-1]), self.fitTimes,
                                   self.partCount)
            self.kOffVar3 = popt[0]
            func = exact_solution(self.partCount[0], self.partCount[-1])
            self.fitSolVar3 = [func(t, popt[0]) for t in self.fitTimes]

        '''
        Variant 4: fits solution of DE to data, fitting off rate and N(Inf),
        leaving N(0) fixed at experimental value
        '''
        if variant == 4:
            def exact_solution(count0):
                def curried_exact_solution(times, koff, countInf):
                    return ((count0 - countInf) *
                            np.exp(-koff * count0 / countInf * times) + countInf)

                return curried_exact_solution

            kOffStart, countInf = 0.005, 50
            popt, pcov = curve_fit(exact_solution(self.partCount[0]), self.fitTimes, self.partCount,
                                   [kOffStart, countInf])
            self.kOffVar4 = popt[0]
            func = exact_solution(self.partCount[0])
            self.fitSolVar4 = [func(t, popt[0], popt[1]) for t in self.fitTimes]

        '''
        Variant 5 (according to supplement Robin et al. 2014):
        Includes cytoplasmic depletion, fixes N(0). N corresponds to R,
        Y corresponds to cytoplasmic volume
        '''
        if variant == 5:
            def exact_solution(count0):
                def curried_exact_solution(times, r1, r2, kPh):
                    return (count0 * ((kPh + r2) / (r2 - r1) * np.exp(r1 * times) +
                                      (kPh + r1) / (r1 - r2) * np.exp(r2 * times)))

                return curried_exact_solution

            popt, pcov = curve_fit(exact_solution(self.partCount[0]), self.fitTimes, self.partCount,
                                   [-0.006, -0.1, 0.1], maxfev=10000)

            self.kPhVar5 = popt[2]
            self.kOnVar5 = (popt[0] * popt[1]) / self.kPhVar5
            self.kOffVar5 = -(popt[0] + popt[1]) - (self.kOnVar5 + self.kPhVar5)
            func = exact_solution(self.partCount[0])
            self.fitSolVar5 = [func(t, popt[0], popt[1], popt[2]) for t in self.fitTimes]

        '''
        Variant 6: This tries to circumvent the error made by fixing the
        starting condition to the first measurement. This point already has a
        statistical error affiliated with it. Fixing it propagates this
        error through the other parameters/the whole fit. Otherwise
        equivalent to variant 5.
        '''
        if variant == 6:
            def curried_exact_solution(times, r1, r2, kPh, count0):
                return (count0 * ((kPh + r2) / (r2 - r1) * np.exp(r1 * times) +
                                  (kPh + r1) / (r1 - r2) * np.exp(r2 * times)))

            popt, pcov = curve_fit(curried_exact_solution, self.fitTimes, self.partCount, [-0.1, -0.2, -0.01, 200],
                                   maxfev=10000)
            self.count0Var6 = popt[3]
            self.kPhVar6 = popt[2]
            self.kOnVar6 = (popt[0] * popt[1]) / self.kPhVar6
            self.kOffVar6 = -(popt[0] + popt[1]) - (self.kOnVar6 + self.kPhVar6)
            self.fitSolVar6 = [curried_exact_solution(t, popt[0], popt[1], self.kPhVar6, self.count0Var6) for t in
                               self.fitTimes]

    def plot_off_rate_fit(self, variant):
        font = {'weight': 'bold', 'size': 'larger'}
        fig, ax = plt.subplots()
        # fig.suptitle('Variant ' + str(variant), fontsize=20, fontdict=font, bbox=dict(facecolor='green', alpha=0.3))
        if variant == 1:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar1)
            ax.set_title(self.kOffVar1)
            print(self.kOffVar1)
        elif variant == 2:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar2)
            ax.set_title(self.kOffVar2)
            print(self.kOffVar2)
        elif variant == 3:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar3)
            ax.set_title(self.kOffVar3)
            print(self.kOffVar3)
        elif variant == 4:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar4)
            ax.set_title(self.kOffVar4)
            print(self.kOffVar4)
        elif variant == 5:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar5)
            ax.set_title(self.kOffVar5)
            print(self.kOffVar5)
        elif variant == 6:
            ax.plot(self.fitTimes, self.partCount, self.fitTimes,
                    self.fitSolVar6)
            ax.set_title(self.kOffVar6)
            print(self.kOffVar6)
        else:
            print('Variant ' + str(variant) + ' does not exist.')
        ax.set(ylabel='# particles', xlabel='t [s]')


"""
ROI

"""


def interp_coors(coors, periodic=True):
    """
    Interpolates coordinates to one pixel distances (or as close as possible to one pixel)
    Linear interpolation

    :param coors:
    :return:
    """

    if periodic:
        c = np.append(coors, [coors[0, :]], axis=0)
    else:
        c = coors
    distances = ((np.diff(c[:, 0]) ** 2) + (np.diff(c[:, 1]) ** 2)) ** 0.5
    distances_cumsum = np.append([0], np.cumsum(distances))
    px = sum(distances) / round(sum(distances))  # effective pixel size
    newpoints = np.zeros((int(round(sum(distances))), 2))
    newcoors_distances_cumsum = 0

    for d in range(int(round(sum(distances)))):
        index = sum(distances_cumsum - newcoors_distances_cumsum <= 0) - 1
        newpoints[d, :] = (
            coors[index, :] + ((newcoors_distances_cumsum - distances_cumsum[index]) / distances[index]) * (
                c[index + 1, :] - c[index, :]))
        newcoors_distances_cumsum += px

    return newpoints


def fit_spline(coors, periodic=True):
    """
    Fits a spline to points specifying the coordinates of the cortex, then interpolates to pixel distances

    :param coors:
    :return:
    """

    # Append the starting x,y coordinates
    if periodic:
        x = np.r_[coors[:, 0], coors[0, 0]]
        y = np.r_[coors[:, 1], coors[0, 1]]
    else:
        x = coors[:, 0]
        y = coors[:, 1]

    # Fit spline
    tck, u = splprep([x, y], s=0, per=periodic)

    # Evaluate spline
    xi, yi = splev(np.linspace(0, 1, 1000), tck)

    # Interpolate
    return interp_coors(np.vstack((xi, yi)).T, periodic=periodic)


class ROI:
    def __init__(self, spline):

        # Inputs
        self.spline = spline
        self.fig = plt.gcf()
        self.ax = plt.gca()

        # Internal
        self._point0 = None
        self._points = None
        self._line = None
        self._fitted = False
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.ax.text(0.03, 0.88,
                     'Specify ROI clockwise from the posterior (4 points minimum)'
                     '\nBACKSPACE: undo'
                     '\nENTER: Proceed',
                     color='white',
                     transform=self.ax.transAxes, fontsize=8)

        # Outputs
        self.xpoints = []
        self.ypoints = []
        self.roi = None

        plt.show(block=True)

    def button_press_callback(self, event):
        if not self._fitted:
            if event.inaxes:
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
                coors = np.vstack((self.xpoints, self.ypoints)).T

                # Spline
                if self.spline:
                    self.roi = fit_spline(coors, periodic=True)

                # Linear interpolation
                else:
                    self.roi = interp_coors(coors, periodic=True)

                # Display line
                self._line = self.ax.plot(self.roi[:, 0], self.roi[:, 1], c='b')
                self.fig.canvas.draw()

                self._fitted = True
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
            self._points = self.ax.scatter(self.xpoints, self.ypoints, c='b')
            self._point0 = self.ax.scatter(self.xpoints[0], self.ypoints[0], c='r')


def def_ROI(img, spline=True):
    """
    Instructions:
    - click to lay down points
    - backspace at any time to remove last point
    - press enter to select area (if spline=True will fit spline to points, otherwise will fit straight lines)
    - at this point can press backspace to go back to laying points
    - press enter again to close and return ROI

    :param img: input image (2d)
    :param spline: if true, fits spline to inputted coordinates
    :return: cell boundary coordinates
    """

    plt.imshow(img, cmap='gray', vmin=0)
    roi = ROI(spline)
    coors = roi.roi
    return coors


"""
Plotting

"""


class PlotCalibration:
    """
    Plot tracked particles

    """

    def __init__(self, frames, threshold=250, featSize=5, maxSize=None):

        # Data
        self.frames = frames
        self.threshold = threshold
        self.featSize = featSize
        self.maxSize = maxSize

        # Set up figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Frame slider
        self.axframe = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=0, valfmt='%d')

        # Threshold slider
        self.axframe2 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe2 = Slider(self.axframe2, 'Threshold', 0, 1000, valinit=self.threshold, valfmt='%d')

        # Featsize slider
        self.axframe3 = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.sframe3 = Slider(self.axframe3, 'Feat Size', 0, 10, valinit=self.featSize, valfmt='%d')

        # Plot parameters
        self.display_points = True
        self.current_frame = 0

        # Display
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.sframe.on_changed(lambda frame: self.update(frame, self.threshold, self.featSize))
        self.sframe2.on_changed(lambda threshold: self.update(self.current_frame, threshold, self.featSize))
        self.sframe3.on_changed(lambda featsize: self.update(self.current_frame, self.threshold, featsize))
        self.update(0, self.threshold, self.featSize)
        self.fig.set_size_inches(10, 20)
        plt.show()

    def key_press_callback(self, event):
        if event.key == 'p':
            self.display_points = not self.display_points
            self.update(self.current_frame, self.threshold, self.featSize)
            self.fig.canvas.draw()

        if event.key == ',':
            self.current_frame -= 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(lambda frame: self.update(frame, self.threshold, self.featSize))
            self.update(self.current_frame, self.threshold, self.featSize)
            self.fig.canvas.draw()

        if event.key == '.':
            self.current_frame += 1
            self.axframe.clear()
            self.sframe = Slider(self.axframe, 'Frame', 0, len(self.frames), valinit=self.current_frame, valfmt='%d')
            self.sframe.on_changed(lambda frame: self.update(frame, self.threshold, self.featSize))
            self.update(self.current_frame, self.threshold, self.featSize)
            self.fig.canvas.draw()

    def update(self, current_frame, threshold, featsize):
        self.current_frame = current_frame
        self.threshold = threshold
        self.featSize = int(featsize) // 2 * 2 + 1

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.frames[int(current_frame)], cmap='gray')

        if self.display_points:
            f = tp.locate(self.frames[int(current_frame)], diameter=self.featSize, invert=False, minmass=self.threshold,
                          maxsize=self.maxSize)
            self.ax.scatter(f['x'], f['y'], facecolors='none', edgecolor='r', s=100, linewidths=1)

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

# def plot_msd(im, c):
#     plt.plot(im.index, im, 'k-', alpha=0.4)  # already in sec
