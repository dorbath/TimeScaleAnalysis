#!/usr/bin/env python3

"""
This script presents the final analysis step: dynamical content
The timescale spectrum of each observable is loaded
and used to derive and plot the dynamical content.
"""

__author__ = "Emanuel Dorbath"

import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
import timescaleanalysis.utils as utils
import timescaleanalysis.plotting as plotting
import timescaleanalysis.io as io
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
    help='Path to timescale spectrum file',
)
@click.option(
    '--output-path',
    '-o',
    'output_path',
    required=True,
    type=click.Path(),
    help='Path to output files',
)
def main(data_path, output_path):
    # Once the analysis is performed, all spectra can be directly reloaded
    # and the dynamical content can be derived
    loaded_spectrum = io.load_npArray(data_path)
    temp_tau_k, temp_dyn_cont = utils.derive_dynamical_content(loaded_spectrum)

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
            'Dynamical content derived from time scale spectra\n'
            'Columns: time, D(tau_k)\n'
        )
    )


if __name__ == '__main__':
    main()
