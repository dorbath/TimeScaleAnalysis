"""Tests for preprocessing module

"""

import numpy as np
import pytest
from pathlib import Path

import timescaleanalysis
import timescaleanalysis.preprocessing

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'


# Test function for preprocessing.generate_input_trajectories
@pytest.mark.parametrize(
    'data_dir, result_directories', [
        (
            'test_data/test_trajectories/test_1ObservableTraj.txt',
            ['test_data/test_trajectories/test_1ObservableTraj.txt']
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


# Test function for preprocessing.load_trajectories
@pytest.mark.parametrize(
    'input_directories, n_traj_conc, averaged, result_files, error', [
        (
            [TEST_TRAJ/'test_1ObservableTraj.txt'],
            None,
            False,
            [TEST_DATA/'test_data_mean/test_data_mean_1ObservableTraj.txt',
             TEST_DATA/'test_data_sem/test_data_sem_1ObservableTraj.txt'],
            None
        ),
        (
            [TEST_TRAJ/'test_3Observables_traj1.txt',
             TEST_TRAJ/'test_3Observables_traj2.txt',
             TEST_TRAJ/'test_3Observables_traj3.txt'],
            None,
            False,
            [TEST_DATA/'test_data_mean/test_data_mean_3Observables.txt',
             TEST_DATA/'test_data_sem/test_data_sem_3Observables.txt'],
            None
        ),
        (
            [TEST_TRAJ/'test_concTraj.txt'],
            3,
            False,
            [TEST_DATA/'test_data_mean/test_data_mean_concTraj.txt',
             TEST_DATA/'test_data_sem/test_data_sem_concTraj.txt'],
            None,
        ),
        (
            [TEST_TRAJ/'test_avgTraj.txt'],
            None,
            True,
            [TEST_DATA/'test_data_mean/test_data_mean_avgTraj.txt',
             TEST_DATA/'test_data_sem/test_data_sem_avgTraj.txt'],
            None
        ),
        (
            [TEST_TRAJ/'test_1ObservableTraj.txt'],
            1,
            True,
            [TEST_DATA/'test_data_mean/test_data_mean_1ObservableTraj.txt',
             TEST_DATA/'test_data_sem/test_data_sem_1ObservableTraj.txt'],
            Exception
        ),
        (
            [TEST_TRAJ/'non_existing_file.txt'],
            None,
            False,
            [TEST_DATA/'test_data_mean/non_existing_file.txt',
             TEST_DATA/'test_data_sem/non_existing_file.txt'],
            FileNotFoundError
        )
    ]
)
def test_load_trajectories(
        input_directories,
        n_traj_conc,
        averaged,
        result_files,
        error):
    preP = timescaleanalysis.preprocessing.Preprocessing(TEST_TRAJ)
    preP.input_directories = input_directories
    if not error:
        preP.load_trajectories(n_traj_conc=n_traj_conc, averaged=averaged)
        expected_mean = np.loadtxt(result_files[0])
        expected_sem = np.loadtxt(result_files[1])
        np.testing.assert_allclose(preP.data_mean, expected_mean)
        np.testing.assert_allclose(preP.data_sem, expected_sem)
    else:
        with pytest.raises(error):
            preP.load_trajectories(n_traj_conc=n_traj_conc, averaged=averaged)


def test_reshape_same_length():
    pass


def test_get_time_array():
    pass


def test_save_preprocessed_data():
    pass
