#!/usr/bin/env python3

"""Example script to perform a timescale analysis for
experimental absorption data.
"""

__author__ = "Emanuel Dorbath"

import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
import timescaleanalysis.utils as utils
import timescaleanalysis.plotting as plotting
import timescaleanalysis.io as io
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
    help='Path to trajectory file with experimental data',
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
def main(data_path, fit_n_decades, output_path):
    """
    This script shows exemplarily how to perform and handle
    experimental data for the timescale analysis.

    The data should have the structure:
    First row: observable names (the value in the first column is ignored)
    First column: time values
    All other values: time traces of the observables

    If the data is not of this format, you can overwrite the important
    parameters in Preprocessing (or TimeScaleAnalysis) instead of using
    load_absorption_spectra, e.g.:
    >>> preP.data_mean = np.loadtxt('path_to_data_mean.txt')
    >>> preP.data_sem = np.loadtxt('path_to_data_sem.txt')
    >>> ...
    However, this may result in errors if the format is not correct or
    relevant data is missing!
    It is recommend to rather transform your input data into the desired shape
    in order to avoid problems!
    """

    ###########################################################################
    # Perform preprocessing
    preP = Preprocessing(data_path)
    preP.generate_input_trajectories()
    preP.load_absorption_spectra()
    preP.save_preprocessed_data(output_path=output_path)
    ###########################################################################

    # Alternatively, directly put data_path into the TSA class
    tsa = TimeScaleAnalysis(preP.data_dir, fit_n_decades)
    tsa.load_data()
    tsa.times *= 1e9  # convert time to ns

    # Derive dynamical content for all observables on the fly
    dynamic_content_arr = np.zeros(
        (tsa.fit_n_decades*10+1)*2,
        dtype=np.float64).reshape((2, (tsa.fit_n_decades*10+1))).T

    ###########################################################################
    # Perform timescale analysis for each observable and plot the results.
    store_spectrum = []  # list that is filled which the amplitudes
    for idxObs in range(tsa.data_mean.shape[1]):
        temp_mean = tsa.data_mean[:, idxObs]
        temp_sem = tsa.data_sem[:, idxObs]
        temp_label = tsa.labels[idxObs]

        # Provide single observable to TSA class
        tsa.options['temp_mean'] = temp_mean
        tsa.options['temp_sem'] = temp_sem
        regPara = 30
        lag_rates = tsa.perform_tsa(
            regPara=regPara,
            startTime=1e-2,
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
        ax1.set_xlim(1e-1, 1e5)
        ax1.set_xlabel(r'$t/\tau_k$ [ns]')
        ax1.set_ylabel(f'{plotting.pretty_label(temp_label, prefix='r')}(t)')
        plotting.save_fig(f'{output_path}/timescale_analysis_{temp_label}.pdf')

        store_time = tsa.spectrum[:, 0]
        store_spectrum.append(tsa.spectrum[:, 1])

        # Derive dynamical content on the fly
        dynamic_content_arr = np.add(dynamic_content_arr, tsa.spectrum**2)
    ###########################################################################

    # Store timescale spectra of each observable for post-analyses
    # This is especially useful if multiple dynamical contents are compared
    io.save_npArray(
        np.column_stack([store_time] + store_spectrum),
        output_path,
        'timescale_spectra',
        comment=(
            'Time scale spectra of all observables\n'
            'Columns:\n'
            'time '+''.join(tsa.labels)+'\n'
            f'Regularization parameter lambda={regPara}, '
            f'fit parameters={tsa.fit_n_decades*10+1}'
        )
    )

    ###########################################################################
    # Once the analysis is performed, all spectra can be directly reloaded
    # and the dynamical content can be derived
    loaded_spectrum = io.load_npArray(f'{output_path}/timescale_spectra')
    temp_tau_k, temp_dyn_cont = utils.derive_dynamical_content(loaded_spectrum)
    ###########################################################################

    ###########################################################################
    # In some cases, multiple dynamical contents are compared (e.g. different
    # selection of observables, different regularization parameters etc.).
    # In this case, multiple 'timescale_spectra' must be loaded and then can
    # be easily plotted into the same figure with the ax=ax1 parameter.
    ax1 = plotting.plot_dynamical_content(temp_tau_k, temp_dyn_cont)
    ax1.set_xlim(1e-1, 1e5)
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
    ###########################################################################


if __name__ == '__main__':
    main()
