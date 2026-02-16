import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from scipy.ndimage import gaussian_filter
import prettypyplot as pplt
import timescaleanalysis.utils as utils
from skimage.filters import threshold_otsu, threshold_multiotsu
from scipy.optimize import curve_fit


def plot_TSA(data_points, data_meanstd, spectrum, times, lag_rates, n_steps):
    """Plot for each distance the averaged time trace and timescale spectrum"""
    upper_bound = np.add(data_points, data_meanstd)
    lower_bound = np.subtract(data_points, data_meanstd)
    laplace_trafo = np.array([np.sum(spectrum[:,1]*np.exp(-times[j]*lag_rates)) for j in range(n_steps)])

    # ax1 plots the time trace and the Laplace transformation
    fig, ax1 = plt.subplots()
    ax1.fill_between(times, lower_bound, upper_bound, lw=0, color='k', alpha=0.4)
    ax1.plot(times, data_points, marker='.', ms=0, lw=1.3, color='k')
    ax1.plot(times, laplace_trafo, marker='.', ms=0, lw=1.0, color='tab:red')
    # ax2 plots the amplitude spectrum
    ax2 = ax1.twinx()
    ax2.plot(spectrum[1:,0], -spectrum[1:,1], marker='.', color='tab:blue', ms=2.5, lw=0.5, ls='--')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.hlines(0, ax1.get_xlim()[0], ax1.get_xlim()[1], colors='k', lw=0.7, ls='--')

    _log_axis(ax1, axis='x')
    ax1.grid(False, axis='y')
    ax2.grid(False)
    ax2.set_yticks([])
    return ax1, ax2


def plot_2D_energy_landscape(dx, dy, n_bins=500):
    """Generate a 2D energy landscape for the quantities (dx,dy)"""
    hist, xbins, ybins, im = plt.hist2d(dx.flatten(), dy.flatten(), bins=n_bins)
    plt.close()

    C = np.zeros((len(xbins)-1, len(ybins)-1))
    max_val_hist = np.amax(hist)
    hist[hist==0] = np.nan
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            C[j][i] = np.log(max_val_hist) - np.log(hist.T[i][j])
    X, Y = np.meshgrid(xbins, ybins)
    C_masked = np.ma.masked_where(np.isnan(C.T), C.T)
    new_cmap = get_alpha_cmap('macaw_r', alpha_fraction=0.3)
    C_mesh = plt.pcolormesh(X, Y, C_masked, ec=None, lw=0, cmap=new_cmap, rasterized=True, zorder=0)
    ax = plt.gca()
    ax.tick_params(direction='in', which='both', zorder=10)
    ax.set_xlabel(r'$C1$')
    ax.set_ylabel(r'$C6$')
    #ax.set_xlim(0.25, 1.1)
    #ax.set_ylim(0.25, 1.1)
    pplt.colorbar(C_mesh, ax=ax, position='right', label=r'$\Delta$G [k${\rm_B}$T]')
    return X, Y, C_masked


def plot_value_heatmaps(dataframes, col, times, output_path=None):
    """
    Plot time-dependent heatmaps for a list of DataFrames, with the y-axis representing value ranges.

    Parameters
    ----------
    dataframes: list of pandas.DataFrame
    col: str, column name of distance
    times: array, time array corresponding to the data
    output_path: str, path to save the figures
    """
    n_t_bins = 51
    n_bins_per_decade = (n_t_bins -1)/5
    time_bins = np.zeros(n_t_bins, dtype=np.float64)
    for k in range(1, n_t_bins):
        time_bins[k] = 10**((1-k)/n_bins_per_decade)

    time_bins[1:] = 0.1/time_bins[1:]
    time_bins[0] = 0
    time_bins = np.append(time_bins, 1e4)
    value_bins = np.arange(0.2, 1.1, 1/100)

    time_bin_count = len(time_bins) - 1  # Number of time bins
    value_bin_count = len(value_bins) - 1  # Number of value bins

    # Combine column data across all DataFrames
    combined_data = np.vstack([df for df in dataframes])

    # Initialize the 2D histogram for probabilities
    heatmap = np.zeros((value_bin_count, time_bin_count))  # Shape: (value_bins, time_bins)

    for i in range(combined_data.shape[0]):  # Loop over all traces
        trace = combined_data[i]
        for t_bin in range(time_bin_count):
            # Find the values for this time bin
            time_start, time_end = time_bins[t_bin], time_bins[t_bin + 1]
            if time_end >= 1e4:
                time_end = 1.1e4 # ensure that last frame is taken
            #time_indices = (times >= time_start) & (times < time_end)
            time_indices = (np.around(times, decimals=6) > time_start) & (np.around(times, decimals=6) <= time_end)
            values_in_bin = trace[time_indices]

            # Bin the values into value_bins
            hist, bbins = np.histogram(values_in_bin, bins=value_bins)
            heatmap[:, t_bin] += hist

    # Normalize along the y-axis (value bins) for probabilities
    heatmap /= heatmap.sum(axis=0, keepdims=True)
    ## Smooth the heatmap using Gaussian smoothing (circumvent nan problem)
    heatmap[np.isnan(heatmap)] = 0.0
    heatmap = gaussian_filter(heatmap, sigma=2, mode='nearest')

    # Plot the heatmap
    cmap = get_alpha_cmap('macaw_r', alpha_fraction=0.25)
    cmap.set_under(color='w')
    fig = plt.figure()
    plt.pcolormesh(time_bins*1e-9, value_bins, heatmap, shading='auto',
                                cmap=cmap, linewidth=0, rasterized=True)
    #plt.hlines(0.45, 1e-10, 1e-5, colors='r', ls='--', lw=1.0)
    conc_combined_data = np.concatenate(combined_data)
    cut_values_threshold = threshold_multiotsu(conc_combined_data[~np.isnan(conc_combined_data)], classes=6)
    print(col, cut_values_threshold)
    plt.hlines(cut_values_threshold, 1e-10, 1e-5, colors='r', ls='--', lw=1.0)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(f'r({col.replace("_",",")}) [nm]')
    plt.xscale('symlog', subs=[2,3,4,5,6,7,8,9], linthresh=1e-11)
    plt.xlim(1e-10, 1e-5)
    plt.ylim(0.2, 1.1)
    plt.yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    save_fig(f'{output_path}/time_dependent_distribution_{col}.pdf')


def fit_log_periodic_oscillations(
        log_time_trace: np.array,
        times: np.array,
        fit_range: list,
        popt: tuple = None):
    """Fit the time trace with a power law (t^a0) and
    logarithmic oscillations with period tau_log.
    A system with hierarchical dynamics may yield equidistant timescales/peaks
    in the timescale spectrum (on a log-time axis).
    This will result in logarithmic oscillations superimposed by a power law
    which describes the diffusive dynamics.

    The fit function is

            sa + sb*t^a0 + sc*t^a0 * cos(2pi/tau log10(t) + phi)

    Parameters
    ----------
    log_time_trace: np.array, data points to fit
            (log-spaced and averaged time trace)
    times: np.array, log-spaced time
    fit_range: list, lower and upper boundary of log-fit in [ns]
            (this improves the fit as many time traces converge at some point)
    popt: tuple, initial guess of all 6 parameters: (a0, tau, s_a, s_b, s_c, phi)
            a0: exponent of power law
            tau: period of logarithmic oscillations
            s_a: offset of fit function
            s_b: amplitude of power law
            s_c: amplitude of logarithmic oscillations
            phi: phase shift of logarithmic oscillations
    filename: str, name of file to save fit parameters (default: None, no saving)
    Return
    ------
    ax1: matplotlib.axes, main plot axis
    ax_insert: matplotlib.axes, inset plot axis
    fitParameters: tuple, optimized fit parameters (a0, tau, s_a, s_b, s_c, phi)
            with their standard deviations from the covariance matrix
    """

    def _fit_log_osc(x, a0, tau, sa, sb, sc, phi): 
        """Full fit function with power law and logarithmic oscillations"""
        return (sa
                + sb*np.power(x, a0)
                + sc*np.power(x, a0)*np.cos((2*np.pi/tau)*np.log10(x)+phi))

    if any(log_time_trace < 0):
        warnings.warn(
            "Negative values in 'log_time_trace'! "
            "This may lead to problems in the fit and plotting. "
            "It is recommended to shift the time trace to positive values.",
            category=Warning,
        )

    fig, ax1 = plt.subplots()
    ax1.tick_params(direction='in', which='both')

    lower_bound = np.where(times >= fit_range[0])[0][0]
    upper_bound = np.where(times <= fit_range[1])[0][-1]

    # Prevent devision by zero problems
    times[np.isclose(times, 0)] = 1e-10

    fit_times = times[lower_bound:upper_bound]
    fit_time_trace = log_time_trace[lower_bound:upper_bound]
    if popt is None:
        a0 = 0.5  # diffusive process
        tau_log = 2.0  # 1/2 oscillations per decade
        sa = fit_time_trace[0]
        sb = 1.0
        sc = 1.0
        phi = -np.pi/2
        warnings.warn(
            "No initial guess for fit parameters provided! "
            "This may lead to problems in the log oscillation fit. "
            "Initial values are guess as follows: "
            f"a0={a0}, tau_log={tau_log}, "
            f"sa={sa:.4f}, "
            f"sb={sb}, sc={sc}, phi={phi:.4f}.",
            category=Warning,
        )
        popt = (a0, tau_log, sa, sb, sc, phi)

    popt, pcov = curve_fit(
        _fit_log_osc,
        fit_times,
        fit_time_trace,
        p0=popt,
        maxfev=10000
    )

    ax1.plot(times, log_time_trace, label=r'$\langle r(t)\rangle$ [nm]', c='k')
    ax1.plot(fit_times, _fit_log_osc(fit_times, *popt), c='tab:red')
    ax1.grid(False, axis='x')
    _log_axis(ax1, axis='xy')
    plt.tick_params(direction='in', which='both')

    ax_insert = ax1.inset_axes([0.5, 0.05, 0.45, 0.45])
    ax_insert.plot(
        times,
        ((log_time_trace - popt[2]) / np.power(times, popt[0]) - popt[3]),
        color='k'
    )
    ax_insert.plot(
        fit_times,
        (_fit_log_osc(fit_times, *popt)-popt[2])
        / np.power(fit_times, popt[0])-popt[3],
        color='tab:red')
    ax_insert.grid(None)
    ax_insert.set_xlim(ax1.get_xlim()[0], ax1.get_xlim()[1])
    ax_insert.set_ylim(-2.5*popt[4], 2.5*popt[4])
    _log_axis(ax_insert, axis='x')
    ax_insert.yaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.xaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.tick_params(labelsize=0, length=2)
    ax_insert.tick_params(direction='in', which='both', labelsize=8)

    return ax1, ax_insert, np.vstack([popt, np.sqrt(np.diag(pcov))])


def get_alpha_cmap(cmap, alpha_fraction=0.1):
    """Add alpha channel to cmap."""
    cmap = plt.get_cmap(cmap)
    cmap_alpha = cmap(np.arange(cmap.N))
    ncolors = len(cmap_alpha) 
    
    alpha = np.ones(ncolors)
    alpha_n = int(alpha_fraction * ncolors)
    alpha[:alpha_n] = np.linspace(0, 1, alpha_n) #elim high values
    #alpha[-alpha_n:] = np.linspace(1, 0, alpha_n) #elim low values
    cmap_alpha[:, -1] = alpha
    return clrs.ListedColormap(cmap_alpha)


def pretty_label(label, prefix='d'):
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


def _log_axis(ax, axis, subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=0.01):
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


def save_fig(path):
    """Save plot in path and print out path for easier access"""
    pplt.hide_empty_axes()
    pplt.savefig(path, bbox_inches='tight')
    print(path)
    plt.close()


def _define_color_cycle():
    """Define new color cycle for better visibility of lines in plots."""
    # TODO: maybe use the one from particle physics or a combination of both
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_colors[0] = '#545454'
    default_colors[1] = '#c05b19'
    default_colors[2] = '#008d66'
    default_colors[3] = '#96ce60'
    default_colors[4] = '#57b5e9'
    default_colors[5] = '#000000'
    default_colors[6] = '#cc00eb'
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=default_colors)
