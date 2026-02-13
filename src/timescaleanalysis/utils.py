import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import prettypyplot as pplt
import timescaleanalysis.plotting as plotting
import os
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu, threshold_multiotsu


def gaussian_smooth(data, sigma, mode='nearest'):
    """Perform Gaussian smoothing/filter

    Parameters
    ---------- 
    data: np.array (1D), data to be smoothed
    sigma: float, standard deviation for Gaussian kernel
    mode: str, behavior at the boundaries of the array
    
    Return
    ------
    smoothed data as np.array
    """

    return gaussian_filter1d(data, sigma, mode=mode)


def save_npArray(array, folder_path, file_name, comment=''):
    """Store a np.array in 'file_path/file_name'

    Parameters
    ---------- 
    array: np.array, 1D or 2D array to be saved
    folder_path: str, path to folder in which file is stored
    file_name: str, name of file,
    comment: str, add description of file to header
    """

    os.makedirs(folder_path, exist_ok=True)  # create the folder if it does not exist yet
    file_path = os.path.join(folder_path, f'{file_name}')
    np.savetxt(file_path, array, header=comment)
    print(f'Saved: {file_path}')


def fit_log_log_time_trace(log_time_trace, times, fit_range, popt=None):
    """Fit the time trace with a power law (a0) and logarithmic oscillations tau_log
    The fit function is sa + sb*t^a0 + sc*t^a0 * cos(2pi/tau log10(t) + phi)

    Parameters
    ----------
    log_time_trace: np.array, data points to fit
    times: np.array, log-spaced time
    fit_range: list, lower and upper boundary of log-fit in [ns]
    popt: tuple, 6 entries: (a0, tau, s_a, s_b, s_c, phi)

    Return
    ------
    ax1: matplotlib.axes, main plot axis
    ax_insert: matplotlib.axes, inset plot axis
    """

    def func_full_fit_log_cos(x, a0, tau, sa, sb, sc, phi): 
        """Full fit function with power law and logarithmic oscillations"""
        return sa + sb*np.power(x,a0) + sc*np.power(x,a0)*np.cos((2*np.pi/tau)*np.log10(x)+phi)

    fig, ax1 = plt.subplots()
    ax1.tick_params(direction='in', which='both')

    lower_bound = np.where(times*1e9>=fit_range[0])[0][0]
    upper_bound = np.where(times*1e9<=fit_range[1])[0][-1]

    fit_times = times[lower_bound:upper_bound]
    fit_time_trace = log_time_trace[lower_bound:upper_bound]
    if popt is None:
        popt, pcov = curve_fit(func_full_fit_log_cos, fit_times*1e9,
                               fit_time_trace, p0=(0.4, 0.97, 0.1, 10, -8, -2.7), maxfev=10000)
    else:
        popt, pcov = curve_fit(func_full_fit_log_cos, fit_times*1e9,
                               fit_time_trace, p0=popt, maxfev=10000)
    for idxP, p in enumerate(popt):
        print(np.around(p,4),np.around(np.sqrt(np.diag(pcov))[idxP],5))

    ax1.plot(times*10**9, log_time_trace, label=r'$x(t)$', c='k')
    ax1.plot(fit_times*10**9, func_full_fit_log_cos(fit_times*10**9, *popt), c='tab:red')
    ax1.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=0.01)
    ax1.set_yscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=0.01)
    ax1.set_xlim(1e-1, 1e4)
    ax1.set_ylim(0.01, 0.4)
    ax1.set_ylabel(r'$\langle\Delta d\rangle$ [nm]')
    ax1.set_xlabel(r'time [ns]')
    plt.tick_params(direction='in', which='both')

    ax_insert = fig.add_axes([0.56,0.47,0.281,0.21])
    ax_insert.plot(times[::10]*10**9, (log_time_trace[::10]-popt[2])/
                   np.power(times[::10]*10**9, popt[0])-popt[3], c='k')
    ax_insert.plot(fit_times*10**9, (func_full_fit_log_cos(fit_times*10**9,
                        *popt)-popt[2])/np.power(fit_times*10**9,popt[0])-popt[3],
                                        c='tab:red', label='fit')
    ax_insert.grid(None)
    ax_insert.set_xlim(ax1.get_xlim()[0], ax1.get_xlim()[1])
    ax_range = np.amax( (func_full_fit_log_cos(fit_times*10**9,
                        *popt)-popt[2])/np.power(fit_times*10**9,popt[0])-popt[3])
    ax_insert.set_ylim(-2.5*popt[4], 2.5*popt[4]) 
    ax_insert.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=0.0001)
    ax_insert.yaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.xaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.tick_params(labelsize=0,length=2)
    ax_insert.tick_params(direction='in', which='both', labelsize=8)
    return ax1, ax_insert


def calculate_ensemble_average_change(data, abs_val=True):
    """Derive the ensemble average change of a given set of distances (e.g. a cluster)
    
    Parameters
    ----------
    data: np.array, data which is used to derive ensemble average change of specific column
    column_name: str, define the name of the column
    n_steps: int, number of steps in data
    abs_val: boolean, selects if absolute difference is derived (default:True)
    
    Return
    ------
    ensemble_averaged_change: np.array, time trace of averaged change
    ensemble_averaged_error: np.array, time trace of corresponding standard deviation
    """

    def _derive_ensemble_averaged_change(time_trace, abs_val=False):
        time_trace_change = time_trace - time_trace[0]
        return time_trace_change if not abs_val else np.abs(time_trace_change)

    # Derive ensemble averaged change
    # Delta d(t) = <d(t) - d(0)>, with d(t)=1/M sum |d_ij(t)| or d(t)=1/M |sum d_ij(t)|
    # Interpretation: derive d(t)-d(0) for each distance and sum them (or sum|X|)
    if data.ndim != 1:
        # Averaging already performed (second column is var/std)
        mean_data = data[:, 0]
    else:
        mean_data = data
    temp_averaged_change = _derive_ensemble_averaged_change(mean_data, abs_val=abs_val)

    return temp_averaged_change


def generate_multi_exp_timetrace(offset, timescales, amplitude, n_steps, sigma=None):
    """Derive time trace from timescale spectrum via a multi-exponential function:
    S(t) = s_0-sum_{k=1,K} s_k e^{-t/tau_k}
    with amplitude s_k and timescales tau_k.

    Parameters
    ----------
    offset: float, s_0 in the multi-exp function
    timescales [ns]: np.array, position of timescales tau_k (log-spaced)
    amplitude: np.array, size of each of the timescales
    n_steps: int, length of exp function
    sigma: float, standard deviation for Gaussian noise rugging the data (default: None, no noise)
    
    Return
    ------
    multiExpFunc: np.array, reconstructed multi-exponential function (with optional noise)"""

    assert len(timescales) == len(amplitude), '"timescales" and "amplitude" must be of same size!'

    times = np.arange(n_steps)
    multiExpFunc = np.full(n_steps, offset, dtype=np.float64)
    for k in range(len(timescales)):
        exp_val = times / timescales[k]
        multiExpFunc -= amplitude[k]*np.exp(-exp_val)
    if sigma is not None:
        data_points = multiExpFunc + np.random.normal(0, sigma, size=multiExpFunc.shape)
    
    plt.plot(times, data_points, c='k', lw=0, marker='o', markersize=0.1)
    plt.plot(times, multiExpFunc, c='tab:red')
    plt.xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=1)
    plotting.save_fig('multi_exp_function_example.pdf')
    save_npArray(np.column_stack(data_points), '.', 'multi_exp_function_example.txt', 
                 comment=f'Multi-exponential function with noise\n Columns: time [ns], S(t) [nm]\n Parameters: offset={offset}, timescales={timescales}, amplitude={amplitude}, sigma={sigma}')

    return data_points


def _reset_class_obj(class_obj):
    """Reset some parameters of the class to avoid conflicts in the loop due to the applied log-spacing"""
    class_obj.quant_data_arr = None
    for data in class_obj.data_arr:
        class_obj.n_steps = len(data) if len(data) > class_obj.n_steps else class_obj.n_steps