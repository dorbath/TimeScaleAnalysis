from genericpath import isfile
import numpy as np
import matplotlib.pyplot as plt
import timescaleanalysis.plotting as plotting
import timescaleanalysis.io as io
import os
import json
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
        in_json_file: str,
        output_path: str = None,
        output_file: str = None):
    """Derive a time trace from a preset timescale spectrum
    via a multi-exponential function:
        S(t) = s_0-sum_{k=1,K} s_k e^{-t/tau_k}
    with amplitude s_k and timescales tau_k.

    This can be very help full to understand the behaviour of
    the multi-exponential function and to test the TSA on a known
    timescales to reproduce the timescale spectrum.

    Parameters
    ----------
    in_json_file: str, path to json file with parameters for multi-exp function
        These parameters are:
            offset: s_0 in the multi-exp function
            timescales: list of positions of timescales tau_k (log-spaced)
            amplitude: list of size of each of the timescales
            n_steps: length of exp function
            sigma: standard deviation for Gaussian noise rugging the data

        Multiple observables can be generated into a single file
        by providing lists for each parameter.
    output_path: str, path to output directory (default: current directory)
    output_file: str, name of output file with generated time traces
            (default: 'multi_exp_function_example.txt')

    Return
    ------
    multiExpFunc: np.array, reconstructed multi-exponential function

    Example:
    --------
    >>> # Generate json file with such a structure:
    >>> # Safe it as '/path/to/json.json'
    >>> {"offset": [1.7, 1.2, 4.4],
    ...  "timescales": [ [1e1, 1e2, 1e4], [3e1, 7e3], [1e2, 3e3, 1e4] ],
    ...  "amplitude": [ [0.2, 0.5, 1.0], [0.5, 0.5], [0.7, 0.9, 2.3] ],
    ...  "n_steps": 2e5,
    ...  "sigma": [0.01, 0.005, 0.02]
    ... }
    >>> generate_multi_exp_timetrace(
    ...     '/path/to/json.json',
    ...     output_path='/path/to/output',
    ...     output_file='output_data.txt')
    """

    def _single_time_trace(
            s_offset: float,
            s_timescales: np.array,
            s_amplitude: np.array,
            s_n_steps: int,
            s_sigma: float = None):
        """Generate time trace for a single observable"""

        assert len(s_timescales) == len(s_amplitude), (
            '"s_timescales" and "s_amplitude" must be of same size!'
        )

        times = np.arange(s_n_steps)
        multiExpFunc = np.full(s_n_steps, s_offset, dtype=np.float64)
        for k in range(len(s_timescales)):
            exp_val = times / s_timescales[k]
            multiExpFunc -= s_amplitude[k]*np.exp(-exp_val)

        if s_sigma is not None:
            generated_data = multiExpFunc + np.random.normal(
                0, s_sigma, size=multiExpFunc.shape
            )
        else:
            generated_data = multiExpFunc

        return generated_data

    # Get n_observables from shape of timescales/offset
    with open(in_json_file, 'r') as f:
        generate_params = json.load(f)
    for key in ['offset', 'timescales', 'amplitude', 'n_steps', 'sigma']:
        if key not in generate_params.keys():
            raise KeyError(f"Expected data file to contain '{key}' key!")

    offset = generate_params['offset']
    timescales = generate_params['timescales']
    amplitude = generate_params['amplitude']
    n_steps = generate_params['n_steps']
    sigma = generate_params['sigma']

    arr_lengths = [len(arr) for arr in [offset, timescales, amplitude, sigma]]
    if len(set(arr_lengths)) != 1:
        raise ValueError(
            "'offset', 'timescales', 'amplitude', and 'sigma' "
            f"must be of same size! Check {in_json_file}."
        )

    n_observables = len(offset)
    data_points = np.full((np.max(n_steps).astype(int), n_observables),
                          None,
                          dtype=np.float32)

    # Generate a time trace for each observable
    for n in range(n_observables):
        data_points[:int(n_steps), n] = _single_time_trace(
            offset[n], timescales[n], amplitude[n], int(n_steps), sigma[n]
        )

    # Make a good header with all relevant information
    output_header = [
        f"Observable {n+1}: "
        f"offset={offset[n]}, timescales={timescales[n]}, "
        f"amplitude={amplitude[n]}, sigma={sigma[n]}\n"
        for n in range(n_observables)
    ]
    if output_path is None:
        output_path = "."
    if output_file is None:
        output_file = "multi_exp_function_example.txt"

    io.save_npArray(
        data_points,
        output_path,
        output_file,
        comment=(
            f"Multi-exponential function with noise\n"
            f"Each row corresponds to a time step (n_steps={int(n_steps)}\n"
            f"Columns: observables S_n(t) [nm]\n"
            f"Parameters of each observable: \n"
            f"{''.join(output_header)}")
            )
    return data_points
