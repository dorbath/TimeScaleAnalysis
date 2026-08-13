import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import prettypyplot as pplt


def plot_TSA(
        data_mean: np.array,
        data_sem: np.array,
        spectrum: np.array,
        times: np.array) -> tuple:
    """Plot for single observable the averaged time trace, the
    corresponding standard error of the mean and timescale spectrum
    obtained from the timescale analysis.
    
    Parameters
    ----------
    data_mean: np.array,
        Mean values of the time trace used for TSA fit
    data_sem: np.array,
        Standard error of the mean values of the time trace
    spectrum: np.array,
        Timescale spectrum with
            1st column: times tau_k
            2nd column: amplitudes s_n of observable
    times: np.array,
        Log-spaced times values
    """
    if spectrum.ndim != 2 or spectrum.shape[1] < 2:
        raise ValueError(
            "Spectrum must have at least two columns: "
            "1st column: times tau_k, "
            "2nd column: amplitudes s_n for each observable"
        )
    if times.ndim != 1:
        raise ValueError("Time array must be a 1D array.")
    if data_mean.shape != data_sem.shape:
        raise ValueError(
            "Mean and SEM arrays must have the same shape. "
            f"Got {data_mean.shape} and {data_sem.shape}."
        )
    if data_mean.shape[0] != times.shape[0]:
        raise ValueError(
            "Mean and SEM arrays must have the same length as the time array. "
            f"Got {data_mean.shape[0]} and {times.shape[0]}."
        )

    n_steps = len(times)
    upper_bound = np.add(data_mean, data_sem)
    lower_bound = np.subtract(data_mean, data_sem)
    lag_rates = np.zeros(spectrum.shape[0])
    lag_rates[1:] = 1.0/spectrum[1:, 0]

    laplace_trafo = np.array(
        [np.sum(spectrum[:, 1]*np.exp(-times[j]*lag_rates))
         for j in range(n_steps)])

    # ax1 plots the time trace and the Laplace transformation
    fig, ax1 = plt.subplots()
    ax1.fill_between(times, lower_bound, upper_bound,
                     lw=0, color='k', alpha=0.4)
    ax1.plot(times, data_mean, marker='.', ms=0, lw=1.3, color='k', label='data')
    ax1.plot(times, laplace_trafo, marker='.', ms=0, lw=1.0, color='tab:red', label='TSA fit')

    # ax2 shows the amplitude spectrum
    ax2 = ax1.twinx()
    ax2.plot(spectrum[1:, 0], -spectrum[1:, 1],
             marker='.', color='tab:blue', ms=2.5, lw=0.5, ls='--', label='TSA spectrum')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.hlines(0, ax1.get_xlim()[0], ax1.get_xlim()[1],
               colors='k', lw=0.7, ls='--')

    _log_axis(ax1, axis='x')
    ax1.grid(False, axis='y')
    ax2.grid(False)
    ax2.set_yticks([])
    pplt.legend(outside='top', axs=[ax1, ax2], ax=ax1, fontsize=6)
    return ax1, ax2


def plot_dynamical_content(
        times: np.array,
        dynamic_content: np.array,
        ax: mpl.axes = None) -> mpl.axes:
    """Plot dynamical content D(tau_k) = sum_n s_n^2.
    The dynamical content is a single observable that describes
    the full behavior of all observables, weighted by their amplitudes.

    Parameters
    ----------
    times: np.array,
        Log-spaced times corresponding to tau_k
    dynamic_content: np.array,
        Dynamical content D(tau_k)
    ax: matplotlib.axes, default=None
        Axis to plot on. This is only important if
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


def plot_2D_histogram(xVal, yVal, zVals) -> None:
    """Plot 2D histogram of two observables.

    Parameters
    ----------
    xVal: np.array,
        Edges of bins for x-axis
    yVal: np.array,
        Edges of bins for y-axis
    zVals: np.array,
        2D array of values for each bin defined by xVal and yVal
    """
    if len(xVal) != zVals.shape[1]+1 or len(yVal) != zVals.shape[0]+1:
        raise ValueError(
            "xVal and yVal are bin edges. "
            "Must have one more entry than zVals.shape[1]"
        )
    # Plot the heatmap
    cmap = get_alpha_cmap('macaw_r', alpha_fraction=0.25)
    cmap.set_under(color='w')
    plt.pcolormesh(xVal, yVal, zVals, shading='auto',
                   cmap=cmap, linewidth=0, rasterized=True)
    _log_axis(plt.gca(), 'x')


def get_alpha_cmap(
        cmap: str,
        alpha_fraction: float = 0.1) -> mpl.colors.ListedColormap:
    """Add alpha channel to cmap for better contrast
    The lower color range will be suppressed, large alpha values suppress
    more of the value range.
    """
    cmap = plt.get_cmap(cmap)
    cmap_alpha = cmap(np.arange(cmap.N))
    ncolors = len(cmap_alpha)

    alpha = np.ones(ncolors)
    alpha_n = int(alpha_fraction * ncolors)
    alpha[:alpha_n] = np.linspace(0, 1, alpha_n)  # suppress low values
    cmap_alpha[:, -1] = alpha
    return mpl.colors.ListedColormap(cmap_alpha)


def pretty_label(label: str, prefix: str = 'd') -> str:
    """Make y-axis label prettier for scientific plotting.

    In many cases, the observable is a distance or angle with the label
    being stored as X_Y (e.g. atoms X,Y)

    Parameters
    ----------
    label: str,
        Label to be made more scientific
    prefix: str, default='d'
        Prefix to be added to label (e.g. 'd' for distance)

    Return
    ------
    Stylized label
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
        linthresh: float = 0.01) -> None:
    """Transform axis to logarithmic scale

    Parameters
    ----------
    ax: matplotlib.axes,
        Plotting axis to be transformed
    axis: str,
        'x' or 'y', defines which axis is transformed
    subs: list, default=[2, 3, 4, 5, 6, 7, 8, 9]
        Ticks that are shown
    linthresh: float, default=0.01
        Threshold for linear region around zero
    """
    if axis == 'xy' or axis == 'yx':
        ax.set_xscale('symlog', subs=subs, linthresh=linthresh)
        ax.set_yscale('symlog', subs=subs, linthresh=linthresh)
    elif axis == 'x':
        ax.set_xscale('symlog', subs=subs, linthresh=linthresh)
    elif axis == 'y':
        ax.set_yscale('symlog', subs=subs, linthresh=linthresh)
    else:
        raise ValueError('Invalid axis! "axis" must be "x" or "y".')


def save_fig(path: str) -> None:
    """Save generated plot in 'path' and print out 'path' for easier access"""
    pplt.hide_empty_axes()
    pplt.savefig(path, bbox_inches='tight')
    print(path)
    plt.close()


def _color_cycle() -> None:
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
