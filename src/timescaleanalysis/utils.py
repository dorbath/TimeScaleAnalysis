import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import timescaleanalysis.plotting as plotting
import os
from scipy.ndimage import gaussian_filter1d


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
    # TODO put into i/o
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{file_name}')
    np.savetxt(file_path, array, header=comment)
    print(f'Saved: {file_path}')


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
        data_points = multiExpFunc + np.random.normal(
            0, sigma, size=multiExpFunc.shape
        )

    plt.plot(times, data_points, c='k', lw=0, marker='o', markersize=0.1)
    plt.plot(times, multiExpFunc, c='tab:red')
    plt.xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=1)
    plotting.save_fig('multi_exp_function_example.pdf')
    save_npArray(
        np.column_stack(data_points),
        ".",
        "multi_exp_function_example.txt",
        comment=(
            f"Multi-exponential function with noise\n"
            f"Columns: time [ns], S(t) [nm]\n Parameters: "
            f"offset={offset}, timescales={timescales}, "
            f"amplitude={amplitude}, sigma={sigma}")
            )

    return data_points


def _reset_class_obj(class_obj):
    """Reset some parameters of the class to avoid conflicts in the loop due to the applied log-spacing"""
    class_obj.quant_data_arr = None
    for data in class_obj.data_arr:
        class_obj.n_steps = len(data) if len(data) > class_obj.n_steps else class_obj.n_steps