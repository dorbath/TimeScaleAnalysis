#!/usr/bin/env python3

"""Fundamental script to perform a timescale analysis.
All required steps are outlined, including:
data generation, preprocess, timescale analysis, plotting and
saving of results.

In several intermediate steps, the user can adjust parameters
as they please, e.g. the used labels for plots, time steps etc.
"""

__author__ = "Emanuel Dorbath"

from genericpath import isfile 
import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
import sys
import timescaleanalysis.utils as utils
import timescaleanalysis.plotting as plotting
import timescaleanalysis.io as io
import timescaleanalysis.supplementary_analyses as suppAna
from timescaleanalysis.timescales import TimeScaleAnalysis
from timescaleanalysis.preprocessing import Preprocessing
import click

pplt.use_style(colors='cbf8', cmap='macaw_r')
rc_fonts = {'figure.figsize': (plt.rcParams['figure.figsize'][0]*2/3,
                               plt.rcParams['figure.figsize'][1]*2/3),
            'font.size': 15,
            'font.weight': 'bold'}
plt.rcParams.update(rc_fonts)
plotting._color_cycle()


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
    '--label-file',
    '-label',
    'label_file',
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
def main(data_path, sim_file, label_file, fit_n_decades, output_path):
    """
    ADD HEADER

    Exponential time traces can be generated with utils.generate_multi_exp_timetrace().
    """

    # Generate multi-exponential time traces with perfectly known
    # timescales, amplitudes.
    #utils.generate_multi_exp_timetrace(
    #    'scripts/example_json.json',
    #    output_path='.',
    #    output_file='multi_exp_function_example.txt'
    #)

    # Needed for general testing of script.
    # If you know your input data adjust the steps as you please.
    preP = None
    # Perform preprocessing
    preP = Preprocessing(
        data_path,
        sim_file=sim_file,
        label_file=label_file
    )
    preP.generate_input_trajectories()
    preP.load_trajectories(n_traj_conc=26)
    preP.get_time_array()
    preP.save_preprocessed_data(output_path=output_path)

    # Plot heatmaps of each observable.
    # These are time-dependent population distributions
    # which reveal the collective shift in observables.
    # It may be advantageous to perform scipy.ndimage.gaussian_filter
    # onto the single heatmap prior to plotting.
    heatmaps = suppAna.get_population_heatmaps(
        preP, lowBound=1e2, upBound=1e7, valueRange=[0.2, 1.1]
    )
    for i in range(len(heatmaps[2])):
        plotting.plot_2D_histogram(heatmaps[0], heatmaps[1], heatmaps[2][i])
        plotting.save_fig(f'{output_path}/time_dependent_distribution_{i}.pdf')

    # If input data_path is already preprocessed file, load it directly.
    # The important parameter is Preprocessing.data_dir from which
    # the preprocess data is loaded.
    if preP is None:
        preP = Preprocessing(data_path)
    assert isfile(preP.data_dir), (
        "Input data is not correctly preprocessed! "
        "Preprocessing.save_preprocessed_data() must be used to save "
        "the preprocessed data."
    )

    # Alternatively, directly put data_path into the TSA class
    # >>> TimeScaleAnalysis(data_path, fit_n_decades)
    tsa = TimeScaleAnalysis(preP.data_dir, fit_n_decades)
    tsa.load_data()

    # Derive dynamical content for all observables on the fly
    dynamic_content_arr = np.zeros(
        (tsa.fit_n_decades*10+1)*2,
        dtype=np.float64).reshape((2, (tsa.fit_n_decades*10+1))).T

    tsa.interpolate_data_points(iterations=2)  # interpolate additional data points as mean
    tsa.log_space_data(5000)  # transform linear frames into log ones
    tsa.extend_timeTrace()  # append additional frames for better convergence

    store_spectrum = []  # list that is filled which the amplitudes
    for idxObs in range(tsa.data_mean.shape[1]):
        temp_mean = utils.gaussian_smooth(tsa.data_mean[:, idxObs], 6)
        temp_sem = utils.gaussian_smooth(tsa.data_sem[:, idxObs], 6)
        temp_label = tsa.labels[idxObs]
        # It can be helpful to rescale the data to be more sensitive
        # This is especially the case for small distances and angles

        # TODO put into separate function
        if False:
            "Scan through several regularization parameters to find the best one"
            regPara, P_Bayes = tsa.perform_tsa(regPara=[1,3,5,7,10,20,30,40,50,60,70,80,90,100], startTime=1e-10)
            plt.plot(regPara, P_Bayes, marker='+', ms=2.5, c='k', lw=1.3)
            plt.xscale('symlog', subs=[2,3,4,5,6,7,8,9], linthresh=1e-10)
            utils.save_fig(f'{output_path}/Bayesian_regPara_{idxObs}.pdf')

        # Provide single observable to TSA class
        tsa.options['temp_mean'] = temp_mean
        tsa.options['temp_sem'] = temp_sem
        regPara = 100
        lag_rates = tsa.perform_tsa(
            regPara=regPara,
            startTime=1e2,
            posVal=False
        )
        ax1, ax2 = plotting.plot_TSA(
            temp_mean,
            temp_sem,
            tsa.spectrum,
            tsa.times,
            lag_rates,
            tsa.n_steps
        )
        ax1.set_xlim(1e2, 1e7)
        ax1.set_xlabel(r'$t/\tau_k$ [ns]')
        ax1.set_ylabel(f'{plotting.pretty_label(temp_label, prefix='r')}(t)')
        plotting.save_fig(f'{output_path}/timescale_analysis_{temp_label}.pdf')

        store_time = tsa.spectrum[:, 0]
        store_spectrum.append(tsa.spectrum[:, 1])

        # Derive dynamical content on the fly
        dynamic_content_arr = np.add(dynamic_content_arr, tsa.spectrum**2)

        #######################################################################
        # This part is only of interest for log-periodic oscillation studies  #
        #
        #fit_range = [1e0, 1e5]
        #ax1, ax_insert, fitParameters = suppAna.fit_log_periodic_oscillations(
        #    temp_mean,
        #    tsa.times,
        #    fit_range,
        #    popt=(0.2, 2.0, 0.0, 1.0, 1.0, -2)
        #)
        #ax1.set_xlim(1e-1, 1e6)
        #ax_insert.set_xlim(1e-1, 1e6)
        #ax1.set_xlabel(r'$t/\tau_k$ [ns]')
        #ax1.set_ylabel(f'{temp_label}(t)')
        #plotting.save_fig(f'{output_path}/log_periodic_fit_{idxObs}.pdf')
        #io.save_npArray(
        #    fitParameters,
        #    output_path,
        #    f'log_periodic_{idxObs}_FitParameters.txt',
        #    comment=(
        #        f'Fit parameters of log-periodic oscillation fit for observable {tsa.labels[idxObs]}:\n'
        #        f'First line are the fit parameters, second line their standard error.\n'
        #        f'Columns: a0, tau, sa, sb, sc, phi\n'
        #        f'Fit function: f(t) = sa + sb*t^a0 + sc*t^a0*cos(2pi/tau*log10(t)+phi)\n'
        #        f'Fit range: {fit_range[0]} to {fit_range[1]}.'
        #    )
        #)
        #######################################################################

    io.save_npArray(
        np.column_stack([store_time] + store_spectrum),
        output_path,
        'timescale_spectra',
        comment=(
            f'Time scale spectra of all observables\n'
            f'Columns:\n'
            f'time '+''.join(tsa.labels)+'\n'
            f'Regularization parameter lambda={regPara}, '
            f'fit parameters={tsa.fit_n_decades*10+1}'
        )
    )

    # Once the analysis is performed, all spectra can be directly reloaded
    # and the dynamical content can be derived
    loaded_spectrum = io.load_npArray(output_path, 'timescale_spectra')
    temp_tau_k, temp_dyn_cont = utils.derive_dynamical_content(loaded_spectrum)

    # In some cases, multiple dynamical contents are compared (e.g. different
    # selection of observables, different regularization parameters etc.).
    # In this case, multiple 'timescale_spectra' must be loaded and then can
    # be easily plotted into the same figure with the ax=ax1 parameter.
    ax1 = plotting.plot_dynamical_content(temp_tau_k, temp_dyn_cont)
    ax1.set_xlim(1e2, 1e7)
    ax1.set_ylim(0, ax1.get_ylim()[1])
    plotting.save_fig(f'{output_path}/dynamical_content.pdf')

    io.save_npArray(
        np.column_stack([temp_tau_k, temp_dyn_cont]),
        output_path,
        'dynamical_content',
        comment=(
            f'Dynamical content derived from time scale spectra\n'
            f'Columns: time, D(tau_k)\n'
            f'Regularization parameter lambda={regPara}, '
            f'fit parameters={tsa.fit_n_decades*10+1}'
        )
    )

    sys.exit()

    #plotting.plot_value_heatmaps(tsa.quant_data_arr, col, tsa.times*1e9, output_path=output_path)

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

    ## TODO: Add 2D plot of TSA of each distance vs time



if __name__ == '__main__':
    main()
