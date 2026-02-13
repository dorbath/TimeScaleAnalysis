#!/usr/bin/env python3

"""The dynamical content of PDZ3 contact distances is derived for multiple distances.
This is a new version of 'contactDistance_ensemble_averaged_change.py' in 'python_scripts/' using Classes in order to be used by others
"""

__author__ = "Emanuel Dorbath"

from genericpath import isfile, isdir
import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
import os
import sys
import timescaleanalysis.utils as utils
import timescaleanalysis.plotting as plotting
from timescaleanalysis.timescales import TimeScaleAnalysis
from timescaleanalysis.preprocessing import Preprocessing
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import click

pplt.use_style(colors='cbf8', cmap='macaw_r')
rc_fonts = {'figure.figsize': (plt.rcParams['figure.figsize'][0]*2/3,
                               plt.rcParams['figure.figsize'][1]*2/3),
            'font.size': 15,
            'font.weight': 'bold'}
plt.rcParams.update(rc_fonts)
plotting._define_color_cycle()



@click.command(
    no_args_is_help='-h',
    help='Perform timescale analysis for a dataset',
)
@click.option(
    '--data-path',
    '-dpp',
    'data_path',
    required=True,
    type=click.Path(),
    help='Path to trajectory files with optional prefix of trajectory fiels',
)
@click.option(
    '--sim-file',
    '-simF',
    'sim_file',
    type=click.STRING,
    default=None,
    help='mdp file used to simulate the trajectories',
)
@click.option(
    '--number-decades',
    '-nD',
    'fit_n_decades',
    required=True,
    type=click.INT,
    help='number of decades used in fit',
)
@click.option(
    '--output-path',
    '-o',
    'output_path',
    required=True,
    type=click.Path(),
    help='Path to output files',
)
def main(data_path, sim_file, fit_n_decades, output_path):
    """All of this is so far done to work for the 1D model
    The aim is to have separate files 'TimeScaleAnalysis_X' for the different systems
    that all use the same class but different preprocessing steps

    Example
    -------
    ./ContactDistanceTSA_PDZ3.py -dpp contactDistances_pdz3/contacts_pdz3_combined_1mus_10mus/cluster -mdp contactDistances_pdz3/pdz3_contactDistances_config.mdp -nD 5 -sys PDZ3L -o test_analysis/
    
    All studied clusters are used: contactDistances_pdz3/contacts_pdz3_combined_1mus_10mus/cluster* (otherwise use cluster1_... for a specific file)
    Simulation parameters: contactDistances_pdz3/pdz3_contactDistances_config.mdp (entries 'dt = ' and 'nstxtcout = ' are mandatory)
    Number of decades covered by timescale analysis: 5
    Figures and data are stored in: test_analysis/


    TODO
    Adjust this file to do following
        - remove any loading of data
        - generate 3 artifical peaks and produce data points (with noise)
        - perform fit onto these data points with log-spacing etc.
        - how to handle 'get_simulation_parameters()'?

    Exponential time traces can be generated with utils.generate_multi_exp_timetrace().
    """
    # utils.generate_multi_exp_timetrace(offset=1.7, timescales=[1e1, 1e2, 1e4],
    #                                   amplitude=[0.2, 0.5, 1.0], n_steps=100000, sigma=0.01)
    preP = None
    # Perform preprocessing
    preP = Preprocessing(data_path, sim_file=sim_file)
    preP.generate_input_trajectories()
    preP.load_trajectories()
    preP.get_time_array()
    preP.save_preprocessed_data(output_path=output_path)

    # If input data_path is already preprocessed file, load it directly.
    # The important parameter is Preprocessing.data_dir
    if preP is None:
        preP = Preprocessing(data_path)
    assert isfile(preP.data_dir), "Input data is not correctly preprocessed! Preprocessing.save_preprocessed_data() must be used to save the preprocessed data."

    tsa = TimeScaleAnalysis(preP.data_dir, fit_n_decades)
    tsa.load_data()

    # Derive dynamical content for all observables
    dynamic_content_arr = np.zeros((tsa.fit_n_decades*10+1)*2, dtype=np.float64).reshape((2, (tsa.fit_n_decades*10+1))).T

    tsa.interpolate_data_points(iterations=2)  # interpolate additional data points as mean
    tsa.log_space_data(500)  # transform linear frames into log ones
    tsa.extend_timeTrace()  # append additional frames for better convergence

    for idxObs in range(tsa.data_mean.shape[1]):
        temp_mean = tsa.data_mean[:, idxObs]
        temp_sem = tsa.data_sem[:, idxObs]
        # This is done to accomplish a more precise fit as distances can be rather small in their change
        scaling_factor = 30
        temp_mean *= scaling_factor
        temp_sem *= scaling_factor
        if False:
            "Scan through several regularization parameters to find the best one"
            regPara, P_Bayes = tsa.perform_tsa(regPara=[1,3,5,7,10,20,30,40,50,60,70,80,90,100], startTime=1e-10)
            plt.plot(regPara, P_Bayes, marker='+', ms=2.5, c='k', lw=1.3)
            plt.xscale('symlog', subs=[2,3,4,5,6,7,8,9], linthresh=1e-10)
            utils.save_fig(f'{output_path}/Bayesian_regPara_{idxObs}.pdf')
        # Provide single observable to TSA class
        tsa.options['temp_mean'] = utils.gaussian_smooth(temp_mean, 6)
        tsa.options['temp_sem'] = utils.gaussian_smooth(temp_sem, 6)
        lag_rates = tsa.perform_tsa(
            regPara=1,
            startTime=1e-1,
            posVal=True
        )
        temp_mean /= scaling_factor
        temp_sem /= scaling_factor
        tsa.spectrum[:, 1] /= scaling_factor

        ax1, ax2 = plotting.plot_TSA(temp_mean,
                                     temp_sem,
                                     tsa.spectrum,
                                     tsa.times,
                                     lag_rates,
                                     tsa.n_steps)
        ax1.set_xlim(1e-1, 1e6)
        ax1.set_xlabel(r'$t/\tau_k$ [ns]')
        ax1.set_ylabel(r'$\langle r(t)\rangle$ [nm]')
        plotting.save_fig(f'{output_path}/timescale_analysis_{idxObs}.pdf')

        dynamic_content_arr = np.add(dynamic_content_arr, tsa.spectrum**2)
    sys.exit()


    plotting.plot_value_heatmaps(tsa.quant_data_arr, col, tsa.times*1e9, output_path=output_path)

    #ensemble_averaged_change = []
    ## Derive ensemble average change (for each column/distance in cluster)
    #temp_ensemble_change = utils.calculate_ensemble_average_change(tsa.quant_data_arr.to_numpy(), abs_val=False)
    #cluster_ensemble_change.append(temp_ensemble_change)
    #ensemble_averaged_change.append(np.mean(cluster_ensemble_change, axis=0))
    ## Plot ensemble averaged change
    #for idxE, ens in enumerate(ensemble_averaged_change):
    #    plt.plot(tsa.times[np.where(tsa.times*1e9 <= 1e4)]*1e9, utils.gaussian_smooth(ens, 6), label=f'C{idxE+1}', lw=1.3)
    #plt.gca().set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=0.001)
    #plt.gca().set_xlim(1e-1, 1e4)
    #plt.gca().set_ylim(-0.15, 0.15)
    #pplt.legend(outside='top', ncols=4, fontsize=7)
    #utils.save_fig(f'{output_path}/ensemble_averaged_change_PDZ3.pdf')

    ## Plot dynamical content
    dynamic_content_arr[0, 0] = 0.0
    dynamic_content_arr[1:(tsa.fit_n_decades*10+1), 0] = 1.0 / lag_rates[1:(tsa.fit_n_decades*10+1)]
    dynamic_content_arr[:,1] = np.sqrt(dynamic_content_arr[:,1])
    fig, ax1 = plt.subplots()
    ax1.plot(dynamic_content_arr[1:, 0], dynamic_content_arr[1:, 1], lw=1.3, ms=0, color='tab:blue')
    ax1.tick_params(direction='in', which='major', top=True, right=False)
    ax1.tick_params(direction='in', which='minor', top=True, right=False)
    ax1.set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=1e-10)
    ax1.set_xlabel(r'$\tau_k$ [s]', labelpad=0)
    ax1.set_ylabel(r'$D(\tau_k)$ [nm]', labelpad=0)
    ax1.grid(False, axis='x', which='major')
    ax1.set_xlim(1e-10, 1e-5)
    #ax1.set_ylim(0, 0.11)
    plt.ylim(0, plt.gca().get_ylim()[1])
    utils.save_fig(f'{output_path}/dynamical_content.pdf')

    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idxDyn, dynCont in enumerate(dynamic_content_clusters_arr):
        if str(idxDyn+1) not in cluster_label:
            continue
        dynCont[0, 0] = 0.0
        dynCont[1:(tsa.fit_n_decades*10+1), 0] = 1.0 / lag_rates[1:(tsa.fit_n_decades*10+1)]
        dynCont[:,1] = np.sqrt(dynCont[:,1])
        ## Save dynamical content of each cluster
        utils.save_npArray(dynCont, output_path, f'dynCont_PDZ3_Cluster{cluster_label[idxDyn]}', 
                           comment=f'Dynamical content of PDZ3 Cluster {cluster_label[idxDyn]}: \nregularization parameter lambda=100, fitPara={tsa.fit_n_decades*10+1}')
        plt.plot(dynCont[1:, 0], dynCont[1:, 1], lw=1.3, ms=0, label=f'C{cluster_label[idxDyn]}', color=clr_cycle[idxDyn])

        ## Derive peaks and their width and plot them
        #_plot_main_timescales(dynCont[1:53, :], cluster_label=cluster_label[idxDyn], color=clr_cycle[idxDyn])
        #_plot_main_timescales(dynCont[1:53, :], height=(0, 0.01), color=clr_cycle[idxDyn], alpha=0.5)
    plt.gca().tick_params(direction='in', which='major', top=True, right=True)
    plt.gca().tick_params(direction='in', which='minor', top=True, right=True)
    plt.gca().set_xscale('symlog', subs=[2, 3, 4, 5, 6, 7, 8, 9], linthresh=1e-10)
    plt.gca().set_xlabel(r'$\tau_k$ [s]', labelpad=0)
    plt.gca().set_ylabel(r'$D(\tau_k)$ [nm]', labelpad=0)
    plt.gca().grid(False, axis='y', which='major')
    plt.gca().set_xlim(1e-10, 1e-5)
    pplt.legend(outside='top', ncols=4, fontsize=9)
    plt.ylim(0, plt.gca().get_ylim()[1])
    utils.save_fig(f'{output_path}/dynamical_content_all_clusters.pdf')

    ## TODO: Add 2D plot of TSA of each distance vs time



if __name__ == '__main__':
    main()
