#!/usr/bin/env python3

"""Fundamental script to perform a timescale analysis.
All required steps are outlined, including:
data generation, preprocess, timescale analysis, plotting and
saving of results.

In several intermediate steps, the user can adjust parameters
as they please, e.g. the used labels for plots, time steps etc.
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
    This script presents a minimal set of required steps going from
    multiple trajectory files with several observables to the
    final timescale analysis for each of them.
    """
    ###########################################################################
    # Perform preprocessing
    preP = Preprocessing(
        data_path,
        sim_file=sim_file,
        label_file=label_file
    )
    preP.generate_input_trajectories()
    preP.load_trajectories()
    preP.get_time_array()
    preP.save_preprocessed_data(output_path=output_path)
    ###########################################################################

    tsa = TimeScaleAnalysis(preP.data_dir, fit_n_decades)
    tsa.load_data()

    # Interpolate additional data points as mean values
    tsa.interpolate_data_points(iterations=2)
    # Transform linear frames into logarithmic ones
    tsa.log_space_data(5000)
    # Append additional frames for a better convergence of the fit
    tsa.extend_timeTrace()

    ###########################################################################
    # Perform timescale analysis for each observable and plot the results.
    store_spectrum = []  # list that is filled which the amplitudes
    for idxObs in range(tsa.data_mean.shape[1]):
        temp_mean = utils.gaussian_smooth(tsa.data_mean[:, idxObs], 6)
        temp_sem = utils.gaussian_smooth(tsa.data_sem[:, idxObs], 6)
        temp_label = tsa.labels[idxObs]

        # Provide single observable to TSA class
        tsa.options['temp_mean'] = temp_mean
        tsa.options['temp_sem'] = temp_sem
        regPara = 100
        lag_rates = tsa.perform_tsa(
            regPara=regPara,
            startTime=1e-1,
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


if __name__ == '__main__':
    main()
