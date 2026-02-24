"""Tests for timescaleanalysis module

"""

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
def test_interpolate_data_points():
    pass


# Test function for timescales.log_space_data
def test_log_space_data():
    pass


# Test function for timescales.extend_timeTrace
def test_extend_timeTrace():
    pass


# Test function for timescales.perform_tsa
def test_perform_tsa():
    pass
