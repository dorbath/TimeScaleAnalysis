#!/usr/bin/env python3

"""
This script presents the first analysis step: preprocessing.
The data is loaded, averaged and preprocessed.
At the end of this step, the preprocessed data is saved
in such a way that it can be used for the timescaleanalysis.
"""

__author__ = "Emanuel Dorbath"

import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
import timescaleanalysis.plotting as plotting
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
    '--output-path',
    '-o',
    'output_path',
    required=True,
    type=click.Path(),
    help='Path to output files',
)
def main(data_path, sim_file, label_file, output_path):
    preP = Preprocessing(
        data_path,
        sim_file=sim_file,
        label_file=label_file
    )
    preP.generate_input_trajectories()
    preP.load_trajectories()
    preP.get_time_array()
    preP.save_preprocessed_data(output_path=output_path)


if __name__ == '__main__':
    main()
