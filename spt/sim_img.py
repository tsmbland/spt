from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp


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
