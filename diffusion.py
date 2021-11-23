import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# from spt import *
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy

"""
Diffusion analysis

To do:
- pad tracks with zeros so can run tracks of different length in batch

"""


class DiffusionFitter:
    """

    """

    def __init__(self, trajectories, model=None, pixel_size=0.124, max_lag_time=6,
                 save_path=None, name='diffusion.csv', timestep=0.03, cnn_batch=True):
        # Data
        self.trajectories = trajectories

        # Model
        self.model = model

        # Parameters
        self.timestep = timestep
        self.pixelSize = pixel_size
        self.max_lag_time = max_lag_time
        self.save_path = save_path
        self.name = name
        self.cnn_batch = cnn_batch

        # Convert to pixel units
        self.trajectories['x'] = self.trajectories['x'].apply(lambda x: x * pixel_size)
        self.trajectories['y'] = self.trajectories['y'].apply(lambda x: x * pixel_size)

        # Results
        self.particles = sorted(self.trajectories.particle.unique())
        self.mss = None
        self.d = None
        self.d_restricted = None
        self.alpha_cnn = None
        self.alpha_msd = None
        self.track_lengths = self.trajectories.groupby('particle').frame.count().values
        self.start_frames = self.trajectories.groupby('particle').frame.min().values
        self.mean_masses = self.trajectories.groupby('particle').mass.mean().values

    def run(self):
        self.mss_analysis()
        self.msd_analysis()
        self.msd_analysis_restricted()
        self.cnn_analysis()
        self.save()

    def mss_analysis(self):
        self.mss = mss_analysis(df=self.trajectories)

    def msd_analysis(self):
        self.d, self.alpha_msd = msd_analysis(df=self.trajectories, timestep=self.timestep,
                                              max_lagtime=self.max_lag_time, min_lagtime=1)

    def msd_analysis_restricted(self):
        self.d_restricted = msd_analysis_brownian(df=self.trajectories, timestep=self.timestep,
                                                  max_lagtime=2, min_lagtime=1)

    def cnn_analysis(self):
        if self.model is not None:
            self.alpha_cnn = cnn_analysis(df=self.trajectories, model=self.model, batch=self.cnn_batch)

    def save(self):
        df = pd.DataFrame(data={'particle': self.particles, 'd': self.d, 'd_restricted': self.d_restricted,
                                'alpha_cnn': self.alpha_cnn, 'alpha_msd': self.alpha_msd,
                                'track_length': self.track_lengths, 'start_frame': self.start_frames,
                                'mean_mass': self.mean_masses, 'mean_step_size': self.mss})
        df.to_csv(self.save_path + '/' + self.name)


def emsd_analysis(df, mtl, timestep=0.03, pixel_size=0.124, max_lagtime=6, min_lagtime=1, brownian=False):
    df2 = copy.deepcopy(df)

    # Convert to pixel units
    df2['x'] = df2['x'].apply(lambda x: x * pixel_size)
    df2['y'] = df2['y'].apply(lambda x: x * pixel_size)

    # Filter
    grouped = df2.groupby('particle')
    trajectories = grouped.filter(lambda x: mtl <= x.frame.count())

    # Calculate displacements
    a = tp.emsd(traj=trajectories, mpp=1, fps=1 / timestep, max_lagtime=max_lagtime)
    times = a.index[min_lagtime - 1:]
    vals = a.values[min_lagtime - 1:]

    # Fit diffusion coefficient and anomalous exponent
    if not brownian:
        results = np.polyfit(np.log10(times), np.log10(vals), 1)  # , w=np.sqrt(p)
        d = (10 ** results[1]) / 4
        alpha = results[0]
    else:
        popt, pcov = curve_fit(lambda x, slope: slope * x, times, vals)
        d = popt[0] / 4
        alpha = 1
    return d, alpha


def msd_analysis(df, timestep, max_lagtime, min_lagtime=1):
    """
    MSD analysis for dataframe
    Uses trackpy method

    """

    n_particles = len(df.particle.unique())

    # Calculate displacements
    a = tp.imsd(df, mpp=1, fps=1 / timestep, max_lagtime=max_lagtime)
    times = a.index[min_lagtime - 1:]
    vals = a.values[min_lagtime - 1:]

    # Fit diffusion coefficients and anomalous exponents
    ds_fitted = np.zeros(n_particles)
    alphas_fitted = np.zeros(n_particles)
    for i, p in enumerate(vals.T):
        results = np.polyfit(np.log10(times), np.log10(p), 1)  # , w=np.sqrt(p)
        ds_fitted[i] = (10 ** results[1]) / 4
        alphas_fitted[i] = results[0]
    return ds_fitted, alphas_fitted


def msd_analysis_brownian(df, timestep, max_lagtime=2, min_lagtime=1):
    """
    MSD analysis for dataframe
    Uses trackpy method

    """

    n_particles = len(df.particle.unique())

    # Calculate displacements
    a = tp.imsd(df, mpp=1, fps=1 / timestep, max_lagtime=max_lagtime)
    times = a.index[min_lagtime - 1:]
    vals = a.values[min_lagtime - 1:]

    # Fit diffusion coefficients
    ds_fitted = np.zeros(n_particles)
    for i, p in enumerate(vals.T):
        popt, pcov = curve_fit(lambda x, slope: slope * x, times, p)
        ds_fitted[i] = popt[0] / 4
    return ds_fitted


def mss_analysis(df):
    g = df.groupby('particle')
    return g.apply(lambda p: np.mean(
        np.sqrt((p['x'][1:].values - p['x'][:-1].values) ** 2 + (p['y'][1:].values - p['y'][:-1]) ** 2))).values


def cnn_analysis(df, model, batch=False):
    """
    Analyse dataframe of tracking data using cnn model

    To do:
    - batch mode, padding tracks to uniform length

    """

    if batch:
        tracks = df_to_tracks(df)
        tracks = preprocess_tracks(tracks)
        res = model.predict(tracks).flatten()

    else:
        particle_ids = sorted(df['particle'].unique())
        res = np.zeros(len(particle_ids))
        for i, p in enumerate(particle_ids):
            trajectory = df[df['particle'] == p]
            track = np.dstack([trajectory.x, trajectory.y])
            track = preprocess_tracks(track)
            a = model.predict(track)
            res[i] = a

    return res


def preprocess_tracks(tracks):
    """
    Return normalised differences

    """

    dx = np.diff(tracks[:, :, 0], axis=1)
    dy = np.diff(tracks[:, :, 1], axis=1)
    meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1),
                              axis=-1)
    return np.dstack((dx / meanstep, dy / meanstep))


def combine_dataframes(file_paths):
    df = pd.DataFrame()
    for f in file_paths:
        single_df = pd.read_csv(f)
        single_df['path'] = f
        df = df.append(single_df)
    return df


def split_tracks(df, track_length):
    """
    Split tracks into non-overlapping sub-tracks
    E.g. with track_length = 20, a track with length 50 will be split into 2 separate tracks, with the final 10
    timepoints unused

    Returns a new dataframe with new particle identities


    """

    # Get start frames
    starts = df.groupby('particle').frame.min()

    # Get new particle IDs
    a = df.apply(lambda x: '%s_%03d' % (int(x.particle), int(((x.frame - starts[x.particle]) // track_length))), axis=1)

    # Create new df
    newdf = df.copy()
    newdf['particle'] = a

    # Filter stubs
    newdf = tp.filter_stubs(newdf, track_length)

    return newdf


def timelapse_tracks(df, track_length):
    """

    :param df:
    :param track_length:
    :return:
    """

    # Create new dataframe
    df_new = pd.DataFrame()

    # Particle ids
    particles = df.particle.unique()

    # Group by particle
    grouped = df.groupby('particle')

    # Loop through particles
    for p in particles:
        g = grouped.get_group(p)
        df_single = pd.DataFrame()
        for i in range(len(g.index) - (track_length - 1)):
            new_track = g.iloc[i:i + track_length]
            new_track['particle'] = '%s_%03d' % (int(p), int(i))
            df_single = df_single.append(new_track)
        df_new = df_new.append(df_single)
    return df_new


def df_to_tracks(df):
    """
    Convert pandas dataframe into 3D tracks array, for input into CNN model
    All tracks must be same length
    Apply split_tracks or timelapse_single_track first, with appropriate track length
    Ordered according to particle number

    """

    xs = np.vstack(list(df.groupby('particle').apply(lambda x: (np.array(x.x) - np.array(x.x)[0]))))
    ys = np.vstack(list(df.groupby('particle').apply(lambda x: (np.array(x.y) - np.array(x.y)[0]))))
    return np.dstack([xs, ys])


def df_to_tracks2(df):
    """
    If all different lengths, pad shorter ones with zeros up to max length

    """

    # Find longest track length
    pass
