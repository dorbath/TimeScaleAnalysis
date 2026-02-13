import warnings

import numpy as np
import pandas as pd
import os
import sys
import math
import timescaleanalysis.utils as utils
import types
import json
from numba import jit
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, curve_fit


def derive_tsa_spectrum(data_points, data_err, regPara, nR, times, lag_rates, n_steps, posVal=False, initValues=None, RegParaSearch=False):
    """Perform fit to get spectrum of TSA
    RegParaSearch=True if Bayesian optimization is performed"""
    entropy_guess = (np.amax(data_points)-np.amin(data_points)) / (100*nR) #nonlinear enhancement factor
    N_dof = n_steps - nR

    @jit(nopython=True)
    def constrain_func_posVal(amplitudes):
        """Constrain to use only positive amplitudes"""
        return -amplitudes[1:]

    @jit(nopython=True) 
    def objective_function(coeff, reg_para=regPara):
        """Fit function chi^2 - lambda*S_ent"""
        fit_data = np.array([np.sum(coeff*np.exp(-times[j]*lag_rates)) for j in range(n_steps)])
        chi2 = np.sum(((fit_data - data_points)/data_err)**2)
        entropy = np.sum(np.sqrt(coeff**2 + 4*entropy_guess**2) - 2*entropy_guess - coeff*np.log((np.sqrt(coeff**2 + 4*entropy_guess**2) + coeff) / (2*entropy_guess)))
        return chi2 - reg_para*entropy
     
    if initValues is None:
        initValues = np.zeros(nR)
        initValues[0] = data_points[-1]
    if not posVal:
        ## Fit with large regPara to get initialValues for amplitudes
        fit_result = minimize(objective_function, initValues, args=(1000), tol=1e-3, options={'maxiter': 1e8}).x
        condition = np.abs(initValues[1:]) < np.amax(np.abs(initValues[1:])) * 0.2
        initValues[1:][condition] *= 0.5
        #fit_result = minimize(objective_function, initValues, tol=1e-3, options={'maxiter': 1e8}).x
        fit_result = minimize(objective_function, fit_result, tol=1e-3, options={'maxiter': 1e8}).x
    else:
        ## Fit with large regPara to get initialValues for amplitudes
        fit_result = minimize(objective_function, initValues, args=(1000), tol=1e-3, options={'maxiter': 1e8}).x
        initValues[1:] = np.where(initValues[1:] > 0, 0.2 * initValues[1:], initValues[1:])
        condition = np.abs(initValues[1:]) < np.amax(np.abs(initValues[1:])) * 0.2
        initValues[1:][condition] *= 0.5
        ## Constrain for only positive fit amplitudes
        constraints = {'type': 'ineq', 'fun': constrain_func_posVal}
        #fit_result = minimize(objective_function, initValues, tol=1e-3, options={'maxiter': 1e8}, constraints=constraints).x
        fit_result = minimize(objective_function, initValues, tol=1e-3, options={'maxiter': 1e8}).x  ## PDZ2L (with scaling x20 instead of x30)

    if not RegParaSearch:
        return fit_result
    else:
        P_Bayes = N_dof*np.log(regPara) - objective_function(fit_result)
        return P_Bayes


class TimeScaleAnalysis:
    """This class is used to perform all major analysis steps


    Parameters
    ----------
    data_file: str, directory to preprocessed data file (json format) which contains the mean and sem of the quantity of interest as well as the time array.
    fit_n_decades: int, number of decades that are covered by the timescale analysis

    Attributes
    ----------
    n_steps: int, number of steps in the simulation (of longest trajectory)
    times: np.array, contains each timestep
    quant_data_arr: np.array, used as input array into the timescale analysis.
                    Can be derived as single column of the input data, sum of multiple ones or as specific distance between atoms (Calpha).
                    Alternatively, it can be overwritten for more specific cases (e.g. contact distances).
    spectrum: np.array (2, X), contains in first column log-spaced timesteps (10 per decade) and in the second the timescale amplitudes

    Example:
    --------
    This class is imported into another .py script for which the parameters are defined and transferred
    >>> from timescale_class import TimeScaleAnalysis
    >>> tsa = TimeScaleAnalysis(data_path, mdp_file, fit_n_decades, )
    >>> tsa.load_trajectories()  # load trajectories in 'data_path'
    >>> tsa.reshape_same_length()  # reshape trajectories to same length
    >>> df_cols = ['quantity_of_interest']  # define name of column of input data
    >>> qoi_cols = ['quantity_of_interest']  # select column of interest
    >>> tsa.get_quantity_of_interest(df_cols, qoi_cols)  # extract the quantitiy of interest
    >>> tsa.load_simulation_parameters()  # get time array
    >>> tsa.log_space_data(2500)  # perform log-spacing
    >>> tsa.derive_average()  # derive average over dataset
    >>> tsa.quant_data_arr['quantity_of_interest'] = utils.gaussian_smooth(tsa.quant_data_arr['quantity_of_interest'], 6)  # apply Gaussian filter
    >>> lag_rates = tsa.perform_tsa(regPara=10, posVal=True, startTime=1e-11)  # perform timescale analysis

    A typical call in the terminal for the .py script for Calpha distances (and the derivation of the dynamical content) takes the form
        ./DynamicalContent_PDZ3.py -dpp cartesian_pdz3_Calpha_combined_1mus_10mus/ca_coord_ -mdp pdz3_config.mdp -nD 5 -sys PDZ3 -o test_analysis/
    where the data are all 'ca_coord_*' files in 'cartesian_pdz3_Calpha_combined_1mus_10mus'.

    In case contact distances are studied, it takes the form
        ./ContactDistanceTSA_PDZ3.py -dpp contactDistances_pdz3/contacts_pdz3_combined_1mus_10mus/cluster -mdp contactDistances_pdz3/pdz3_contactDistances_config.mdp -nD 5 -sys PDZ3L -o test_analysis/
    where multiple contact files (here 'cluster*' must have the form clusterX_...) are loaded.
    IMPORTANT: 
    The usage of contact distances requires a column file for each cluster to label them correctly. This file must be located (in this example) at
        contactDistances_pdz3/contacts_pdz3_combined_1mus_10mus/column_names/cX_label'
    where X must match with clusterX_*
    """

    # These kwargs are used to more conviniently perform the fit
    DEFAULTS = {
        "temp_mean": None,
        "temp_sem": None,
    }

    def __init__(self, data_file, fit_n_decades, **kwargs):
        assert os.path.isfile(data_file), f'Expected data_file is not a valid file!'
        self.data_file = data_file
        if not isinstance(fit_n_decades, int):
            warnings.warn(f'Expected "fit_n_decades" to be type int, but got {type(fit_n_decades)}. Conversion may result in round-off errors.')
        self.fit_n_decades = int(fit_n_decades)
        self.data_mean = None
        self.data_sem = None
        self.n_steps = 0
        self.times = None
        self.quant_data_arr = None
        self.spectrum = None

        self.options = self.DEFAULTS | kwargs
    
    def load_data(self):
        """Load preprocessed data with the correct shape"""
        with open(self.data_file) as f:
            data_json = json.load(f)
        assert 'data_mean' in data_json.keys(), "Expected data file to contain 'data_mean' key!"
        assert 'data_sem' in data_json.keys(), "Expected data file to contain 'data_sem' key!"
        assert 'times' in data_json.keys(), "Expected data file to contain 'times' key!"
        self.data_mean = np.array(data_json['data_mean'], dtype=np.float32)
        self.data_sem = np.array(data_json['data_sem'], dtype=np.float32)
        self.times = np.array(data_json['times'], dtype=np.float32)
        self.n_steps = len(self.times)

    def interpolate_data_points(self, iterations: int = 1):
        """Interpolate additional frames as mean between neighboring frames
        
        Parameters
        ----------
        iterations: int, number of times to perform interpolation, each iteration doubles the number of frames"""
        def interpolation_step(interpArr):
            """Interpolate neighboring frames by average"""
            interpArr = np.asarray(interpArr)
            if interpArr.ndim == 1:
                interpArr = interpArr[:, None]
                squeeze_arr = True  # return 1D after interpolation
            elif interpArr.ndim == 2:
                squeeze_arr = False
            else:
                raise ValueError("interpArr must be 1D or 2D for interpolation")

            interpolate = (interpArr[:-1] + interpArr[1:]) / 2
            tempArr = np.empty(
                (interpArr.shape[0] + interpolate.shape[0],
                 interpArr.shape[1]),
                dtype=interpArr.dtype)
            tempArr[0::2] = interpArr
            tempArr[1::2] = interpolate

            return tempArr[:, 0] if squeeze_arr else tempArr
        
        for _ in range(iterations):
            self.times = interpolation_step(self.times)
            self.data_mean = interpolation_step(self.data_mean)
            self.data_sem = interpolation_step(self.data_sem)
            self.n_steps = len(self.times)

    def log_space_data(self, target_n_steps: int):
        """Convert the linear data set into a log-spaced one
        Perfect log-spacing is not possible, since the frames are integers
        and thus for the first few decades the log-spacing is more linear.
        For alter decades the spacing is perfectly log-spaced.
        Reducing 'target_n_steps' or using 'interpolation_step' can improve
        the log-spacing for the first decades.
        
        Parameters
        ----------
        target_n_steps: int, used in np.geomspace, actual number of frames is always below this value
        """
        log_spaced_index_mask = np.unique(
            np.geomspace(1, self.n_steps,
                         num=target_n_steps,
                         dtype=np.int32,
                         endpoint=True))
        if log_spaced_index_mask[-1] == self.n_steps:
            log_spaced_index_mask[-1] = log_spaced_index_mask[-1] - 1  # prevent errors for last index
        log_spaced_index_mask = np.insert(log_spaced_index_mask, 0, 0)  # add first frame to log spacing
        self.data_mean = self.data_mean[log_spaced_index_mask]
        self.data_sem = self.data_sem[log_spaced_index_mask]
        self.times = self.times[log_spaced_index_mask]
        self.n_steps = len(log_spaced_index_mask)

    def extend_timeTrace(self):
        """Append one order of magnitude to the data by a constant value derived as average over ~1/2 decade"""
        # Create copys to avoid overwriting of original data during appending
        times = self.times
        data_mean = np.asarray(self.data_mean)
        data_sem = np.asarray(self.data_sem)

        squeeze_arr = False
        # (N,1) arrays must be adjusted for generalization
        if data_mean.ndim == 1:
            data_mean = data_mean[:, None]
            data_sem = data_sem[:, None]
            squeeze_arr = True

        N, _ = data_mean.shape

        # Get number of frames to append and frame index for average over last half decade
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

        # restore original dimensionality
        if squeeze_arr:
            data_mean = data_mean[:, 0]
            data_sem = data_sem[:, 0]

        self.data_mean = data_mean
        self.data_sem = data_sem
        self.times = times
        self.n_steps = len(times)

    def perform_tsa(self, regPara, startTime=1e0, posVal=False, initValues=None):
        """Perform the timescale analysis S(t) = SUM a_k exp(-t/tau_k)

        Parameters
        ----------
        regPara: float or array, regularization parameter used to weight entropy term
                (if array, a search for optimal regPara is performed using Bayesian criterion)
        startTime: float, fastest timescale used in fit
        powVal: bool, set True to only use positive fit amplitudes
        initValues: np.array, entries for initial guess of amplitudes, if None: zeros are used"""
        # Setup fit
        nR = self.fit_n_decades*10 + 1  # number of fit amplitudes
        lag_rates = np.zeros(nR, dtype=np.float64)
        for k in range(1, nR):
            lag_rates[k] = 1/startTime *10**((1-k)/10)

        if isinstance(regPara, float) or isinstance(regPara, int):
            temp_fit_amplitudes = derive_tsa_spectrum(np.copy(self.options['temp_mean']),
                                                      np.copy(self.options['temp_sem']),
                                                      regPara, nR, self.times, lag_rates, self.n_steps,
                                                      posVal=posVal, initValues=initValues)
        elif isinstance(regPara, np.ndarray) or isinstance(regPara, list):
            print('Performing Bayesian search for optimal regularization parameter')
            regPara = sorted(regPara)
            P_Bayes_arr = []
            for regParaVal in regPara:
                temp_P_Bayes = derive_tsa_spectrum(np.copy(self.quant_data_arr.quantity_of_interest.values),
                                                   np.copy(self.quant_data_arr.quantity_of_interest_meanstd.values),
                                                   regParaVal, nR, self.times, lag_rates, self.n_steps,
                                                   posVal=posVal, initValues=initValues,
                                                   RegParaSearch=True)
                P_Bayes_arr.append(temp_P_Bayes)
            exp_P_Bayes = np.exp(P_Bayes_arr/(np.abs(np.amax(P_Bayes_arr))/10))
            return regPara, exp_P_Bayes/np.amax(exp_P_Bayes)
        else:
            raise TypeError(f'Expected regPara to be float or np.ndarray/list, but got {type(regPara)}')
        self.spectrum = np.zeros((2,nR), dtype=np.float32).T
        self.spectrum[:,1] = temp_fit_amplitudes
        for k in range(1, nR):
            self.spectrum[k][0] = 1.0/lag_rates[k]
        return lag_rates