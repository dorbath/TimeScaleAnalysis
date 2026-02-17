from genericpath import isfile
import numpy as np
import matplotlib.pyplot as plt
import timescaleanalysis.plotting as plotting
import timescaleanalysis.io as io
import os
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth(data: np.array, sigma: float, mode: str = 'nearest'):
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


def generate_input_trajectories(file_dir: str):
    """Get all files/trajectories in 'file_dir' with the correct prefix.
    All files that fulfill file_dir* are taken as input.
    """
    # Isolate folder path and trajectory prefix
    data_dir_split = file_dir.split("/")
    folder_prefix = ''
    folder_suffix = data_dir_split[-1]
    for n in range(len(data_dir_split)-1):
        folder_prefix += data_dir_split[n]+'/'
    # If prefix matches exactly with a file, take this file as input
    # Otherwise take all files with matching prefix
    if isfile(folder_prefix+'/'+folder_suffix):
        input_directories = [
            folder_suffix
        ]
    else:
        input_directories = [
            path for path in os.listdir(folder_prefix)
            if path.startswith(folder_suffix)
        ]
    return folder_prefix, input_directories


def derive_dynamical_content(spectrum: np.array):
    """Derive the dynamical content D(tau_k) = sum_n s_n^2.
    The dynamical content is a single observable that describes
    the full behavior of all observables, weighted by their amplitudes.

    Parameters
    ----------
    spectrum: np.array, timescale spectrum with
            1st column: times tau_k
            All other columns: amplitues s_n for each observable

    Return
    ------
    tau_k: np.array, times corresponding to the timescale spectrum
    dynamic_content: np.array, dynamical content D(tau_k)
    """
    # The first entry is removed as it corresponds to an offset that
    # does not contribute to the dynamics
    print(spectrum.shape)
    if spectrum.shape[1] < 2:
        raise ValueError(
            "Spectrum must have at least two columns: "
            "1st column: times tau_k, "
            "2nd column: amplitudes s_n for each observable"
        )
    tau_k = spectrum[1:, 0]
    dynamic_content = np.sum(spectrum[1:, 1:]**2, axis=1)
    return tau_k, dynamic_content


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


def generate_multi_exp_timetrace(
        offset: float,
        timescales: np.array,
        amplitude: np.array,
        n_steps: int,
        sigma: float = None):
    """Derive a time trace from a preset timescale spectrum
    via a multi-exponential function:
        S(t) = s_0-sum_{k=1,K} s_k e^{-t/tau_k}
    with amplitude s_k and timescales tau_k.

    Parameters
    ----------
    offset: float, s_0 in the multi-exp function
    timescales [ns]: np.array, position of timescales tau_k (log-spaced)
    amplitude: np.array, size of each of the timescales
    n_steps: int, length of exp function
    sigma: float, standard deviation for Gaussian noise rugging the data
            (default: None, no noise)

    Return
    ------
    multiExpFunc: np.array, reconstructed multi-exponential function"""

    assert len(timescales) == len(amplitude), (
        '"timescales" and "amplitude" must be of same size!'
    )

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
    io.save_npArray(
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
