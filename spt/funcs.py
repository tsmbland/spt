from matplotlib.path import Path
import numpy as np
import trackpy as tp
import glob


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
