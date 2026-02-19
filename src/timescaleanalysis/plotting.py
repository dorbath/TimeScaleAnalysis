import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import prettypyplot as pplt
from skimage.filters import threshold_multiotsu


def plot_TSA(
        data_mean: np.array,
        data_sem: np.array,
        spectrum: np.array,
        times: np.array,
        lag_rates: np.array,
        n_steps: int):
    """Plot for each observable the averaged time trace and timescale spectrum"""
    upper_bound = np.add(data_mean, data_sem)
    lower_bound = np.subtract(data_mean, data_sem)
    laplace_trafo = np.array(
        [np.sum(spectrum[:, 1]*np.exp(-times[j]*lag_rates))
         for j in range(n_steps)])

    # ax1 plots the time trace and the Laplace transformation
    fig, ax1 = plt.subplots()
    ax1.fill_between(times, lower_bound, upper_bound,
                     lw=0, color='k', alpha=0.4)
    ax1.plot(times, data_mean, marker='.', ms=0, lw=1.3, color='k')
    ax1.plot(times, laplace_trafo, marker='.', ms=0, lw=1.0, color='tab:red')

    # ax2 shows the amplitude spectrum
    ax2 = ax1.twinx()
    ax2.plot(spectrum[1:, 0], -spectrum[1:, 1],
             marker='.', color='tab:blue', ms=2.5, lw=0.5, ls='--')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.hlines(0, ax1.get_xlim()[0], ax1.get_xlim()[1],
               colors='k', lw=0.7, ls='--')

    _log_axis(ax1, axis='x')
    ax1.grid(False, axis='y')
    ax2.grid(False)
    ax2.set_yticks([])
    return ax1, ax2


def plot_dynamical_content(
        times: np.array,
        dynamic_content: np.array,
        ax: mpl.axes = None):
    """Plot dynamical content D(tau_k) = sum_n s_n^2.
    The dynamical content is a single observable that describes
    the full behavior of all observables, weighted by their amplitudes.

    Parameters
    ----------
    times: np.array, log-spaced times corresponding to tau_k
    dynamic_content: np.array, dynamical content D(tau_k)
    ax: matplotlib.axes, axis to plot on. This is important if
            multiple dynamical contents are plotted in the same figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(times, dynamic_content, lw=1.3, ms=0)
    ax.tick_params(direction='in', which='major', top=True, right=False)
    ax.tick_params(direction='in', which='minor', top=True, right=False)
    _log_axis(ax, axis='x')
    ax.set_xlabel(r'$\tau_k$', labelpad=0)
    ax.set_ylabel(r'$D(\tau_k)$', labelpad=0)
    ax.grid(False, axis='x', which='major')
    return ax


def plot_2D_histogram(xVal, yVal, zVals):
    """Plot 2D histogram of two observables.

    Parameters
    ----------
    xVal: np.array, edges of bins for x-axis
    yVal: np.array, edges of bins for y-axis
    zVals: np.array, 2D array of values for each bin defined by xVal and yVal
    """
    assert len(xVal) == zVals.shape[1]+1, (
        "xVal are bin edges. Must have one more entry than zVals.shape[1]"
    )
    assert len(yVal) == zVals.shape[0]+1, (
        "yVal are bin edges. Must have one more entry than zVals.shape[0]"
    )
    # Plot the heatmap
    cmap = get_alpha_cmap('macaw_r', alpha_fraction=0.25)
    cmap.set_under(color='w')
    plt.pcolormesh(xVal, yVal, zVals, shading='auto',
                   cmap=cmap, linewidth=0, rasterized=True)
    _log_axis(plt.gca(), 'x')


def get_alpha_cmap(cmap: str, alpha_fraction: float = 0.1):
    """Add alpha channel to cmap for better contrast"""
    cmap = plt.get_cmap(cmap)
    cmap_alpha = cmap(np.arange(cmap.N))
    ncolors = len(cmap_alpha)

    alpha = np.ones(ncolors)
    alpha_n = int(alpha_fraction * ncolors)
    alpha[:alpha_n] = np.linspace(0, 1, alpha_n)  # remove high values
    cmap_alpha[:, -1] = alpha
    return mpl.colors.ListedColormap(cmap_alpha)


def pretty_label(label: str, prefix: str = 'd'):
    """Make y-axis label prettier for scientific plotting
    In many cases, the observable is a distance or angle with the label
    being stored as X_Y (e.g. atoms X,Y)

    Parameters
    ----------
    label: str, label to be made more scientific
    prefix: str, prefix to be added to label (e.g. 'd' for distance)
    """
    if '_' in label:
        label = '('+label+')'
    label = prefix + label
    label = label.replace('_', ',')
    return label


def _log_axis(
        ax: mpl.axes,
        axis: str,
        subs: list = [2, 3, 4, 5, 6, 7, 8, 9],
        linthresh: float = 0.01):
    """Transform axis to logarithmic scale

    Parameters
    ----------
    ax: matplotlib.axes, axis to be transformed
    axis: str, 'x' or 'y', defines which axis is transformed"""
    if axis == 'xy' or axis == 'yx':
        ax.set_xscale('symlog', subs=subs, linthresh=linthresh)
        ax.set_yscale('symlog', subs=subs, linthresh=linthresh)
    elif axis == 'x':
        ax.set_xscale('symlog', subs=subs, linthresh=linthresh)
    elif axis == 'y':
        ax.set_yscale('symlog', subs=subs, linthresh=linthresh)
    else:
        raise ValueError('Invalid axis! "axis" must be "x" or "y".')


def save_fig(path: str):
    """Save generated plot in 'path' and print out 'path' for easier access"""
    pplt.hide_empty_axes()
    pplt.savefig(path, bbox_inches='tight')
    print(path)
    plt.close()


def _color_cycle():
    """Color cycle for red-green colorblind friendly plots."""
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_colors[0] = '#005B8E'  # blue
    default_colors[1] = '#E69F00'  # orange
    default_colors[2] = '#D55E00'  # vermillion
    default_colors[3] = '#000000'  # black
    default_colors[4] = '#BE548F'  # purple
    default_colors[5] = '#009E73'  # bluish green
    default_colors[6] = '#56B4E9'  # light blue
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=default_colors)
