"""Tests for utils module

"""

import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import json as json

import timescaleanalysis.utils as utils

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'


# Test function for utils.gaussian_smooth
@pytest.mark.parametrize(
        'data, sigma, result', [
            (
                np.array([0., 1., 2., 3., 4.]),
                1,
                np.array([0.36378461, 1.06312251, 2., 2.93687749, 3.63621539])
            )
        ]
)
def test_gaussian_smooth(
        data,
        sigma,
        result):
    smoothed_data = utils.gaussian_smooth(data, sigma)
    np.testing.assert_allclose(smoothed_data, result)


# Test function for utils.generate_input_trajectories
@pytest.mark.parametrize(
    'file_dir, result_directories', [
        (
            TEST_TRAJ/'test_1ObservableTraj.txt',
            [TEST_TRAJ/'test_1ObservableTraj.txt']
        ),
        (
            TEST_TRAJ/'test_3Observables_traj',
            [TEST_TRAJ/'test_3Observables_traj1.txt',
             TEST_TRAJ/'test_3Observables_traj2.txt',
             TEST_TRAJ/'test_3Observables_traj3.txt']
        ),
        (
            TEST_TRAJ/'non_existing_file.txt',
            []
        ),
    ]
)
def test_generate_input_trajectories(
        file_dir,
        result_directories):
    input_directories = utils.generate_input_trajectories(file_dir)
    assert input_directories == result_directories


def test_derive_dynamical_content():
    pass


def test_absmax():
    pass


def test_generate_multi_exp_timetrace():
    pass
