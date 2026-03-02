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


# Test function for utils.derive_dynamical_content
@pytest.mark.parametrize(
        'input_spectrum, result, error', [
            (
                TEST_DATA/'test_tsa_spectrum/test_spectra_3Observables.txt',
                TEST_DATA/'test_tsa_spectrum/test_dynCont_3Observables.txt',
                None
            ),
            (
                TEST_DATA/'test_tsa_spectrum/test_SingleCol.txt',
                None,
                ValueError
            ),
        ]
)
def test_derive_dynamical_content(
        input_spectrum,
        result,
        error):
    spectrum = np.loadtxt(input_spectrum)
    if not error:
        tau_k, dynamic_content = utils.derive_dynamical_content(spectrum)
        np.testing.assert_allclose(tau_k, np.loadtxt(result)[:, 0])
        np.testing.assert_allclose(dynamic_content, np.loadtxt(result)[:, 1])
    else:
        with pytest.raises(error):
            utils.derive_dynamical_content(spectrum)


# Test function for utils.absmax
@pytest.mark.parametrize(
        'data, axis, result', [
            (
                np.array([[1, -5, 6], [-4, 2, -3]]),
                0,
                np.array([4, 5, 6])
            ),
            (
                np.array([[1, -5, 6], [-4, 2, -3]]),
                1,
                np.array([6, 4])
            ),
            (
                np.array([[1, -5, 6], [-4, 2, -3]]),
                None,
                6
            )
        ]
)
def test_absmax(
        data,
        axis,
        result):
    np.testing.assert_allclose(utils.absmax(data, axis), result)


# Test function for utils.generate_multi_exp_timetrace
@pytest.mark.parametrize(
        'input_json', [
                TEST_DATA/'test_generate_timetrace/test_multiExp.json'
        ]
)
def test_generate_multi_exp_timetrace(
        input_json):
    output_file = TEST_DATA/'test_output_files/test_multiExp_timetrace.txt'
    multiExp_data = utils.generate_multi_exp_timetrace(
        str(input_json),
        str(TEST_DATA/'test_output_files/'),
        'test_multiExp_timetrace.txt'
    )
    assert isfile(output_file)
    with open(input_json, 'r') as f:
        json_dic = json.load(f)
    assert multiExp_data.shape[0] == json_dic['n_steps']
    assert multiExp_data.shape[1] == len(json_dic['offset'])
    np.testing.assert_equal(
        multiExp_data,
        np.loadtxt(output_file))
