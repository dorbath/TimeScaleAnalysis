"Plotting"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from scipy.ndimage import gaussian_filter
import prettypyplot as pplt
from skimage.filters import threshold_otsu, threshold_multiotsu


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

    ax1.set_xscale('symlog', subs=[2,3,4,5,6,7,8,9], linthresh=1e-10)
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
