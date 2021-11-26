from matplotlib.widgets import Slider
import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
import ipywidgets as widgets


def view_stack(stack, jupyter=False):
    if not jupyter:
        view_stack_tk(stack)
    else:
        view_stack_jupyter(stack)


def view_stack_tk(stack, vmin=None, vmax=None):
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


def view_stack_jupyter(stack, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    @widgets.interact(Frame=(0, len(stack) - 1, 1))
    def update(Frame=0):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        ax.imshow(stack[Frame], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylim(*ylim)


def plot_calibration(frames, threshold=250, featSize=5, maxSize=None, separation=None, jupyter=False):
    if not jupyter:
        PlotCalibration(frames, threshold=threshold, featSize=featSize, maxSize=maxSize, separation=separation)
    else:
        plot_calibration_jupyter(frames, threshold=threshold, featSize=featSize, maxSize=maxSize, separation=separation)


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
            # print(np.median(np.array(tp.proximity(f))))

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)


def plot_calibration_jupyter(frames, threshold=250, featSize=5, maxSize=None, separation=None):
    if separation is None:
        separation = featSize + 1

    fig, ax = plt.subplots()
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    @widgets.interact(Frame=(0, len(frames) - 1, 1), threshold=(0, 1000, 10), featSize=(1, 11, 2),
                      separation=(0, 10, 1))
    def update(Frame=0, threshold=threshold, featSize=featSize, separation=separation):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        ax.imshow(frames[Frame], cmap='gray')

        f = tp.locate(frames[Frame], diameter=featSize, invert=False, minmass=threshold,
                      maxsize=maxSize, separation=separation)
        ax.scatter(f['x'], f['y'], facecolors='none', edgecolor='r', s=100, linewidths=1)
        # print(np.median(np.array(tp.proximity(f))))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


def plot_features(frames, trajectories, jupyter=False):
    if not jupyter:
        PlotFeatures(frames, trajectories)
    else:
        plot_features_jupyter(frames, trajectories)


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


def plot_features_jupyter(frames, trajectories):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    @widgets.interact(Frame=(0, len(frames) - 1, 1))
    def update(Frame=0):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        f = trajectories[trajectories.frame == Frame]
        ax.imshow(frames[Frame], cmap='gray')

        cmap = pl.cm.hsv(np.linspace(0, 1, 20))
        for p in f['particle']:
            single_traj = trajectories[trajectories['particle'] == p]
            single_traj = single_traj[single_traj['frame'] <= Frame]
            ax.plot(single_traj['x'], single_traj['y'], c=cmap[(p % 20)], linewidth=0.5)

        a = ax.scatter(f['x'], f['y'], c=(f['particle'] % 20), cmap='hsv', vmin=0, vmax=20, s=100,
                       linewidths=1)
        a.set_facecolor('none')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


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
    fig, ax = plt.subplots()
    unstacked = traj.set_index(['particle', 'frame'])[['x', 'y']].unstack()
    for i, trajectory in unstacked.iterrows():
        if c is not None:
            ax.plot(trajectory['x'], trajectory['y'], c=c)
        else:
            ax.plot(trajectory['x'], trajectory['y'], c=c)
