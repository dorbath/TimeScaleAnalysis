"""Tests for preprocessing module

"""

import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import json as json

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


# Test function for preprocessing.load_absorption_spectra
@pytest.mark.parametrize(
        'input_directories, result', [
            (
                [TEST_DATA/'test_absorption_spectra/test_absorption.txt'],
                TEST_DATA/'test_absorption_spectra/test_json_absorption.json'
            )
        ]
)
def test_load_absorption_spectra(
        input_directories,
        result):
    preP = timescaleanalysis.preprocessing.Preprocessing(TEST_TRAJ)
    preP.input_directories = input_directories
    preP.load_absorption_spectra()
    with open(result, 'r') as f:
        expected_result = json.load(f)
        np.testing.assert_allclose(
            preP.data_mean, expected_result['data_mean']
        )
        np.testing.assert_allclose(
            preP.data_sem, expected_result['data_sem']
        )
        np.testing.assert_allclose(
            preP.options['times'], expected_result['times']
        )
        np.testing.assert_equal(
            preP.labels_lst, expected_result['labels']
        )


# Test function for preprocessing.reshape_same_length
@pytest.mark.parametrize(
    'input_data_arr, n_steps, result_shape, error', [
        (
            [TEST_DATA/'test_data_arr/test_data_arr_1Observable_10thFrame.txt'],
            int(1e4),
            (1, 1e4),
            None
        ),
        (
            [TEST_DATA/'test_data_arr/test_data_arr_1Observable_10thFrame.txt',
             TEST_DATA/'test_data_arr/'
             'test_data_arr_1Observable_short_10thFrame.txt'],
            int(1e4),
            (2, 1e4),
            None
        ),
        (
            [TEST_DATA/'test_data_arr/test_data_arr_1Observable_10thFrame.txt'],
            int(500),
            None,
            ValueError
        ),
    ]
)
def test_reshape_same_length(
        input_data_arr,
        n_steps,
        result_shape,
        error):
    preP = timescaleanalysis.preprocessing.Preprocessing(TEST_TRAJ)
    preP.data_arr = [
        np.loadtxt(data, dtype=np.float16)
        for data in input_data_arr
    ]
    preP.n_steps = n_steps
    if not error:
        preP.reshape_same_length()
        np.testing.assert_equal(np.shape(preP.data_arr), result_shape)
        assert np.asarray(preP.data_arr).ndim == 2
    else:
        with pytest.raises(error):
            preP.reshape_same_length()


# Test function for preprocessing.get_time_array
@pytest.mark.parametrize(
    'sim_file, result, error', [
        (
            TEST_DATA/'test_time_array/test_sim_file.md',
            TEST_DATA/'test_time_array/test_time_array_sim_file.txt',
            None
        ),
        (
            None,
            TEST_DATA/'test_time_array/test_time_array_no_sim_file.txt',
            None
        ),
        (
            TEST_DATA/'test_time_array/non_existing_file.md',
            None,
            FileNotFoundError
        ),
    ]
)
def test_get_time_array(
        sim_file,
        result,
        error):
    preP = timescaleanalysis.preprocessing.Preprocessing(TEST_TRAJ)
    preP.options['sim_file'] = sim_file
    preP.n_steps = int(1e4)
    if not error:
        preP.get_time_array()
        expected_time_array = np.loadtxt(result)
        np.testing.assert_allclose(
            preP.options['times'], expected_time_array
        )
    else:
        with pytest.raises(error):
            preP.get_time_array()


# Test function for preprocessing.save_preprocessed_data
@pytest.mark.parametrize(
    'labels, output_path, result, error', [
        (
            TEST_DATA/'test_save_data/test_labels.txt',
            TEST_DATA/'test_output_files/',
            TEST_DATA/'test_save_data/test_result.json',
            None
        ),
        (
            None,
            TEST_DATA/'test_output_files/',
            TEST_DATA/'test_save_data/test_result_default_labels.json',
            None
        ),
        (
            TEST_DATA/'test_save_data/test_wrong_labels.txt',
            TEST_DATA/'test_output_files/',
            None,
            ValueError
        )
    ]
)
def test_save_preprocessed_data(
        labels,
        output_path,
        result,
        error):
    preP = timescaleanalysis.preprocessing.Preprocessing(TEST_TRAJ)
    preP.data_mean = np.loadtxt(
        TEST_DATA/'test_save_data/test_data_mean.txt'
    )
    preP.data_sem = np.loadtxt(
        TEST_DATA/'test_save_data/test_data_sem.txt'
    )
    preP.options['times'] = np.loadtxt(
        TEST_DATA/'test_save_data/test_time_array.txt'
    )
    preP.options['label_file'] = labels
    if not error:
        preP.save_preprocessed_data(output_path=str(output_path))
        assert isfile(preP.data_dir)
        with open(result, 'r') as f:
            output_data = json.load(f)
        np.testing.assert_allclose(
            preP.data_mean.tolist(), output_data['data_mean']
        )
        np.testing.assert_allclose(
            preP.data_sem.tolist(), output_data['data_sem']
        )
        np.testing.assert_allclose(
            preP.options['times'].tolist(), output_data['times']
        )
        np.testing.assert_equal(
            preP.labels_lst, output_data['labels']
        )
    else:
        with pytest.raises(error):
            preP.save_preprocessed_data(output_path=str(output_path))
