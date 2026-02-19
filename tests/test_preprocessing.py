"""Tests for preprocessing module

"""

import numpy as np
import pytest

import timescaleanalysis
import timescaleanalysis.preprocessing

TEST_FILES = [
    'test_data/test_trajectories/test_1Observable_traj1.txt',
    ['test_data/test_trajectories/test_3Observables_traj1.txt',
     'test_data/test_trajectories/test_3Observables_traj2.txt',
     'test_data/test_trajectories/test_3Observables_traj3.txt'],
    'test_data/test_trajectories/test_avgTraj.txt',
    'test_data/test_trajectories/test_concTraj.txt',
    'test_data/test_trajectories/non_existing_file.txt'
]


@pytest.fixture
def test_files():
    return TEST_FILES


@pytest.mark.parametrize(
    'data_dir, result_directories', [
        (
            'test_data/test_trajectories/test_1Observable_traj1.txt',
            ['test_data/test_trajectories/test_1Observable_traj1.txt']
        ),
        (
            'test_data/test_trajectories/test_3Observables_traj',
            ['test_data/test_trajectories/test_3Observables_traj1.txt',
             'test_data/test_trajectories/test_3Observables_traj2.txt',
             'test_data/test_trajectories/test_3Observables_traj3.txt']
        ),
        (
            'test_data/test_trajectories/non_existing_file.txt',
            []
        ),
    ]
)
def test_generate_input_trajectories(
        data_dir, result_directories):
    preP = timescaleanalysis.preprocessing.Preprocessing(data_dir)
    preP.generate_input_trajectories()
    assert preP.input_directories == result_directories


#@pytest.mark.parametrize(
#    ''
#)
#def test_load_trajectories():
