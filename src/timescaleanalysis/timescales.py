from genericpath import isfile
import warnings

import numpy as np
import json
from numba import jit
from scipy.optimize import minimize


def derive_tsa_spectrum(
        data_points: np.array,
        data_err: np.array,
        times: np.array,
        n_steps: int,
        regPara: float,
        nR: int,
        lag_rates: np.array,
        initValues: np.array = None,
        posVal: bool = False,
        RegParaSearch: bool = False) -> np.array:
    """Perform multi-exponential fit to get spectrum of TSA.
    The fit function has the form: S(t) = SUM s_k exp(-t/tau_k)
    where the amplitudes s_k are fitted for given lag rates (1/tau_k).
    Peaks in s_k(tau_k) reveal dynamical processes at timescale tau_k!

    Parameters
    ----------
    data_points: np.array, data time trace of observable to fit
    data_err: np.array, time trace of standard error of the mean
    times: np.array, time array for fit
    n_steps: int, number of steps in data
    regPara: float or list, regularization parameter weights entropy term
    nR: int, number of fit amplitudes
    lag_rates: np.array, log-spaced array of lag rates (1/tau_k)
    initValues: np.array, entries for initial guess of amplitudes,
                if None: zeros are used (default: None)
    posVal: bool, set True to only use positive fit amplitudes (default: False)
    RegParaSearch: bool, set True to perform Bayesian optimization of regPara
                   (this will return the Bayesian posterior probability),
                   else return fit result for given regPara (default: False)

    Return
    ------
    (if RegParaSearch=False)
    fit_result: np.array, fitted amplitudes for each lag_rate
    (if RegParaSearch=True)
    P_Bayes: float, Bayesian posterior probability for given regPara
    """
    # Nonlinear enhancement factor: see doi.org/10.1366/000370205364
    entropy_guess = (np.amax(data_points)-np.amin(data_points)) / (100*nR)
    N_dof = n_steps - nR

    @jit(nopython=True)
    def constrain_func_posVal(amplitudes):
        """Constrain to use only positive amplitudes"""
        return -amplitudes[1:]

    @jit(nopython=True)
    def objective_function(
            coeff: np.array,
            reg_para: float = regPara):
        """Fit function chi^2 - lambda*S_ent
        'coeff' is the inital guess for the amplitudes s_k
        """
        fit_data = np.array(
            [np.sum(coeff*np.exp(-times[j]*lag_rates)) for j in range(n_steps)]
        )
        chi2 = np.sum(((fit_data - data_points)/data_err)**2)
        temp_sqrt = np.sqrt(coeff**2 + 4*entropy_guess**2)
        entropy = np.sum(
            temp_sqrt
            - 2*entropy_guess
            - coeff * np.log((temp_sqrt + coeff) / (2*entropy_guess))
        )
        return chi2 - reg_para*entropy

    if initValues is None:
        initValues = np.zeros(nR)
        initValues[0] = data_points[-1]

    if not posVal:
        constraints = None
    else:
        constraints = {'type': 'ineq', 'fun': constrain_func_posVal}

    # Fit with large regPara to get good initialValues for amplitudes
    # This imporves the stability of the final fit
    pre_fit = minimize(
        objective_function,
        initValues,
        args=(regPara*50),
        tol=1e-3,
        options={'maxiter': 1e8}
    ).x

    # Suppress small amplitudes to improve convergence of final fit
    condition = np.abs(pre_fit[1:]) < np.amax(np.abs(pre_fit[1:])) * 0.2
    pre_fit[1:][condition] *= 0.2
    if posVal:
        # Reduce negative amplitdues (in fit "-pre_fit" is used)
        pre_fit[1:] = np.where(pre_fit[1:] > 0, 0.2 * pre_fit[1:], pre_fit[1:])

    fit_result = minimize(
        objective_function,
        pre_fit,
        tol=1e-3,
        options={'maxiter': 1e8},
        constraints=constraints
    ).x

    if not RegParaSearch:
        return fit_result
    else:
        P_Bayes = N_dof*np.log(regPara) - objective_function(fit_result)
        return P_Bayes


class TimeScaleAnalysis:
    """This class is used to perform all major analysis steps
    Preprocessed data (mean, sem and time array) must be provided as json file.
        (See preprocessing.py to generate correct file format)

    Parameters
    ----------
    data_file: str, directory to preprocessed data file (json format)
               It contains the mean, sem of all studied observables
               as well as the time array.
    fit_n_decades: int, number of decades that are covered by the
               timescale analysis.
    data_mean: np.array (N, M), contains the time-dependent mean
               of all observables. (N,) not yet supported!
    data_sem: np.array (N, M), contains the time-dependent
              standard error of the mean. (N,) not yet supported!
    n_steps: int, number of steps in the simulation
    times: np.array, contains each timestep
    spectrum: np.array (2, X),
              first column contains log-spaced timesteps (10 per decade)
              second column contains the fitted timescale amplitudes

    Kwargs
    ------
    temp_mean: np.array, mean data of single observable used in fit
    temp_sem: np.array, sem data of single observable used in fit

    Example:
    --------
    >>> from timescale_class import TimeScaleAnalysis
    >>> from preprocessing import Preprocessing
    >>> # Generate several multi-exp function
    >>> utils.generate_multi_exp_timetrace(
    ...         offset=1.7,
    ...         timescales=[1e1, 1e2, 1e4],
    ...         amplitude=[0.2, 0.5, 1.0],
    ...         n_steps=100000,
    ...         sigma=0.01)
    >>> # Stored them at 'path/to/data'
    >>> preP = Preprocessing('path/to/data')
    >>> preP.generate_input_trajectories()
    >>> preP.load_trajectories()
    >>> preP.get_time_array()
    >>> preP.save_preprocessed_data(output_path=output_path)
    >>> # Preprocessing finished. Start with analysis.
    >>> tsa = TimeScaleAnalysis(preP.data_dir, 6)
    >>> tsa.load_trajectories()
    >>> tsa.log_space_data(500)
    >>> for idxObs in range(tsa.data_mean.shape[1]):
    >>>     temp_mean = tsa.data_mean[:, idxObs]
    >>>     temp_sem = tsa.data_sem[:, idxObs]
    >>>     lag_rates = tsa.perform_tsa(
    ...         regPara=100,
    ...         startTime=1e-1,)
    >>> plotting.plot_TSA(temp_mean,
    ...     temp_sem,
    ...     tsa.spectrum,
    ...     tsa.times,
    ...     lag_rates,
    ...     tsa.n_steps)
    >>> plotting.save_fig('tsa_example.pdf')
    """

    # These kwargs are used to more conviniently perform the fit
    DEFAULTS = {
        "temp_mean": None,
        "temp_sem": None,
    }

    def __init__(self,
                 data_file: str,
                 fit_n_decades: int,
                 **kwargs):
        if not isinstance(fit_n_decades, int):
            warnings.warn(
                f"Expected 'fit_n_decades' to be type int, but got "
                f"{type(fit_n_decades)}. Conversion  into 'int' "
                f"may result in round-off errors.",
                category=Warning,
            )
        self.data_file = data_file
        self.fit_n_decades = int(fit_n_decades)

        self.data_mean = None
        self.data_sem = None
        self.n_steps = 0
        self.times = None
        self.labels = None
        self.spectrum = None

        self.options = self.DEFAULTS | kwargs

    def load_data(self) -> None:
        """Load preprocessed data with the correct shape"""
        if not isfile(self.data_file):
            raise FileNotFoundError(
                f"Provided data_file {self.data_file} does not exist!"
            )
        with open(self.data_file) as f:
            data_json = json.load(f)
        for key in ['data_mean', 'data_sem', 'times', 'labels']:
            if key not in data_json.keys():
                raise KeyError(f"Expected data file to contain '{key}' key!")
        self.data_mean = np.array(data_json['data_mean'], dtype=np.float32)
        self.data_sem = np.array(data_json['data_sem'], dtype=np.float32)
        self.times = np.array(data_json['times'], dtype=np.float32)
        self.labels = np.array(data_json['labels'], dtype=str)
        self.n_steps = len(self.times)

        # Catch 1D arrays of shape (N,) and convert them to (N,1)
        if self.data_mean.ndim == 1:
            self.data_mean = np.expand_dims(self.data_mean, axis=1)
            self.data_sem = np.expand_dims(self.data_sem, axis=1)

    def interpolate_data_points(self, iterations: int = 1) -> None:
        """Interpolate additional frames as mean between neighboring frames

        Parameters
        ----------
        iterations: int, number of times to perform interpolation,
                    each iteration doubles the number of frames
        """
        def interpolation_step(interpArr: np.array) -> np.array:
            """Interpolate neighboring frames by average

            Parameters
            ----------
            interpArr: np.array, array to interpolate of shape (N,1) or (N,M)

            Return
            ------
            np.array, interpolated array of shape (2N-1,M)
            """
            interpArr = np.asarray(interpArr)
            if interpArr.ndim == 1:
                interpArr = interpArr[:, None]
                squeeze_arr = True  # return 1D after interpolation
            elif interpArr.ndim == 2:
                squeeze_arr = False
            else:
                raise ValueError(
                    "'interpArr' must be 1D or 2D array for interpolation"
                )

            interpolate = (interpArr[:-1] + interpArr[1:]) / 2
            tempArr = np.empty(
                (interpArr.shape[0] + interpolate.shape[0],
                 interpArr.shape[1]),
                dtype=interpArr.dtype)
            tempArr[0::2] = interpArr
            tempArr[1::2] = interpolate

            return tempArr[:, 0] if squeeze_arr else tempArr

        if not isinstance(iterations, int):
            warnings.warn(
                f"Expected 'iterations' to be type int, but got "
                f"{type(iterations)}. Conversion  into 'int' "
                f"may result in round-off errors.",
                category=Warning,
            )
        iterations = int(iterations)
        for _ in range(iterations):
            self.times = interpolation_step(self.times)
            self.data_mean = interpolation_step(self.data_mean)
            self.data_sem = interpolation_step(self.data_sem)
            self.n_steps = len(self.times)

    def log_space_data(self, target_n_steps: int) -> None:
        """Convert the linear data set into a log-spaced one
        Perfect log-spacing is not possible, since the frames are integers
        and thus for the first few decades the log-spacing is more linear.
        For alter decades the spacing is perfectly log-spaced.
        Reducing 'target_n_steps' or using 'interpolation_step' can improve
        the log-spacing for the first decades.

        Parameters
        ----------
        target_n_steps: int, used in np.geomspace,
                        actual number of frames is always below this value
        """
        log_spaced_index_mask = np.unique(
            np.geomspace(1, self.n_steps,
                         num=target_n_steps,
                         dtype=np.int32,
                         endpoint=True))
        # Prevent errors for last index
        if log_spaced_index_mask[-1] == self.n_steps:
            log_spaced_index_mask[-1] = log_spaced_index_mask[-1] - 1
        # Add first frame to log-spacing and eliminate any duplicates
        log_spaced_index_mask = np.unique(
            np.insert(log_spaced_index_mask, 0, 0)
        )
        self.data_mean = self.data_mean[log_spaced_index_mask]
        self.data_sem = self.data_sem[log_spaced_index_mask]
        self.times = self.times[log_spaced_index_mask]
        self.n_steps = len(log_spaced_index_mask)

    def extend_timeTrace(self) -> None:
        """Append one order of magnitude to the data by a constant value
        derived as average over ~1/2 decade.
        On a log-scale, the final half decade is rather short."""
        # Create copys to avoid overwriting of original data during appending
        times = self.times
        data_mean = np.asarray(self.data_mean)
        data_sem = np.asarray(self.data_sem)

        N, _ = data_mean.shape

        # Get number of frames to append and frame index from which
        # the final half decade begins.
        n_frames_perOrder = N - np.where(times > times[-1]/10)[0][0]
        low_bound_frame = len(np.where(times >= times[-1]/2)[0])

        # Perform averages of mean and SEM
        temp_append_mean = np.mean(data_mean[-low_bound_frame:], axis=0)
        temp_append_sem = np.sqrt(
            np.sum(np.square(data_sem[-low_bound_frame:]), axis=0)
            / low_bound_frame**2
        )

        data_mean = np.concatenate(
            [data_mean, np.tile(temp_append_mean, (n_frames_perOrder, 1))],
            axis=0
        )
        data_sem = np.concatenate(
            [data_sem, np.tile(temp_append_sem, (n_frames_perOrder, 1))],
            axis=0
        )
        times = np.concatenate(
            [times, times[-n_frames_perOrder:] * 10]
        )

        self.data_mean = data_mean
        self.data_sem = data_sem
        self.times = times
        self.n_steps = len(times)

    def perform_tsa(self,
                    regPara: float | list,
                    startTime: float = 1e0,
                    posVal: bool = False,
                    initValues: np.array = None) -> None:
        """Perform the timescale analysis S(t) = SUM a_k exp(-t/tau_k)
        The lag rates (1/tau_k) are log-spaced with 10 points per decade
        and the amplitudes a_k are fitted, revealing dynamical processes.

        Parameters
        ----------
        regPara: float or list, regularization parameter weights entropy term
                (if list, a search for optimal regPara is performed
                using a Bayesian criterion; doi.org/10.1366/0003702077797014)
        startTime: float, fastest timescale used in fit
        powVal: bool, set True to only use positive fit amplitudes
        initValues: np.array, entries for initial guess of amplitudes,
                    if None: zeros are used"""
        # Setup fit
        nR = self.fit_n_decades*10 + 1  # number of fit amplitudes
        lag_rates = np.zeros(nR, dtype=np.float64)
        for k in range(1, nR):
            lag_rates[k] = 1/startTime * 10**((1-k)/10)

        if isinstance(regPara, float) or isinstance(regPara, int):
            temp_fit_amplitudes = derive_tsa_spectrum(
                np.copy(self.options['temp_mean']),
                np.copy(self.options['temp_sem']),
                self.times,
                self.n_steps,
                regPara,
                nR,
                lag_rates,
                initValues=initValues,
                posVal=posVal,
            )
        elif isinstance(regPara, np.ndarray) or isinstance(regPara, list):
            # TODO Separate function with good header
            print("'ragPara' is list:\n Performing Bayesian search\
                  for optimal regularization parameter")
            regPara = sorted(regPara)
            P_Bayes_arr = []
            for regParaVal in regPara:
                temp_P_Bayes = derive_tsa_spectrum(
                    np.copy(self.options['temp_mean']),
                    np.copy(self.options['temp_sem']),
                    self.times,
                    self.n_steps,
                    regParaVal,
                    nR,
                    lag_rates,
                    initValues=initValues,
                    posVal=posVal,
                    RegParaSearch=True,
                )
                P_Bayes_arr.append(temp_P_Bayes)
            exp_P_Bayes = np.exp(P_Bayes_arr/(np.abs(np.amax(P_Bayes_arr))/10))
            return regPara, exp_P_Bayes/np.amax(exp_P_Bayes)
        else:
            raise TypeError(
                f'Expected regPara to be float or np.ndarray/list, '
                f'but got {type(regPara)}'
            )
        self.spectrum = np.zeros((2, nR), dtype=np.float32).T
        self.spectrum[:, 1] = temp_fit_amplitudes
        for k in range(1, nR):
            self.spectrum[k][0] = 1.0/lag_rates[k]
        return lag_rates
