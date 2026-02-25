"""Tests for timescaleanalysis module

"""

from tracemalloc import take_snapshot
import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import json as json

import timescaleanalysis
import timescaleanalysis.timescales

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'

# Test function for timescales.load_data
@pytest.mark.parametrize(
    'json_file, result_mean, result_sem, result_times, result_labels, error', [
        (
            TEST_DATA/'test_save_data/test_result.json',
            TEST_DATA/'test_save_data/test_data_mean.txt',
            TEST_DATA/'test_save_data/test_data_sem.txt',
            TEST_DATA/'test_save_data/test_time_array.txt',
            TEST_DATA/'test_save_data/test_labels.txt',
            None
        ),
        (
            TEST_DATA/'test_data_mean/non_existing_file.txt',
            None,
            None,
            None,
            None,
            FileNotFoundError
        ),
    ]
)
def test_load_data(
        json_file,
        result_mean,
        result_sem,
        result_times,
        result_labels,
        error):
    tsa = timescaleanalysis.timescales.TimeScaleAnalysis(json_file, 1)
    if not error:
        tsa.load_data()
        np.testing.assert_allclose(
            tsa.data_mean, np.loadtxt(result_mean).reshape(tsa.data_mean.shape)
        )
        np.testing.assert_allclose(
            tsa.data_sem, np.loadtxt(result_sem).reshape(tsa.data_sem.shape)
        )
        np.testing.assert_allclose(
            tsa.times, np.loadtxt(result_times).reshape(tsa.times.shape)
        )
        np.testing.assert_array_equal(
            tsa.labels, np.loadtxt(result_labels, dtype=str)
        )
    else:
        with pytest.raises(error):
            tsa.load_data()


# Test function for timescales.interpolate_data_points
@pytest.mark.parametrize(
    'array, iterations, result, error', [
        (
            np.array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]]),
            1,
            np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.],
                      [4., 5.], [5., 6.], [6., 7.]]),
            None
        ),
        (
            np.array([0., 1., 2., 3.]),
            2,
            np.array([0., 0.25, 0.5, 0.75, 1.,
                      1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.]),
            None
        ),
        (
            np.array([
                [[0., 1., 2., 3.], [0., 1., 2., 3.]],
                [[0., 1., 2., 3.], [0., 1., 2., 3.]]
                ]),
            1,
            None,
            ValueError
        ),
    ]
)
def test_interpolate_data_points(
        array,
        iterations,
        result,
        error):
    tsa = timescaleanalysis.timescales.TimeScaleAnalysis(TEST_TRAJ, 1)
    if not error:
        tsa.data_mean = array
        tsa.data_sem = array
        tsa.times = array
        tsa.n_steps = len(tsa.times)
        tsa.interpolate_data_points(iterations)
        np.testing.assert_allclose(tsa.data_mean, result)
    else:
        with pytest.raises(error):
            tsa.interpolate_data_points(iterations)


# Test function for timescales.log_space_data
@pytest.mark.parametrize(
        'lin_shape, log_result', [
            (
                (10000, 1),
                (2125, 1)
            ),
            (
                (10000, 10),
                (2125, 10)
            ),
            (
                (100, 1),
                (100, 1)
            )
        ]
)
def test_log_space_data(
        lin_shape,
        log_result):
    tsa = timescaleanalysis.timescales.TimeScaleAnalysis(TEST_TRAJ, 1)
    tsa.data_mean = np.zeros(lin_shape)
    tsa.data_sem = np.zeros(lin_shape)
    tsa.times = np.arange(lin_shape[0])
    tsa.n_steps = lin_shape[0]
    tsa.log_space_data(target_n_steps=5000)
    assert tsa.data_mean.shape == log_result


# Test function for timescales.extend_timeTrace
@pytest.mark.parametrize(
        'initial_shape, extended_result', [
            (
                (10000, 3),
                (19000, 3)
            ),
            (
                (100, 1),
                (190, 1)
            )
        ]
)
def test_extend_timeTrace(
        initial_shape,
        extended_result):
    tsa = timescaleanalysis.timescales.TimeScaleAnalysis(TEST_TRAJ, 1)
    tsa.data_mean = np.zeros(initial_shape)
    tsa.data_sem = np.zeros(initial_shape)
    tsa.times = np.arange(initial_shape[0])
    tsa.extend_timeTrace()
    assert tsa.data_mean.shape == extended_result


# Test function for timescales.perform_tsa
def test_perform_tsa():
    pass
