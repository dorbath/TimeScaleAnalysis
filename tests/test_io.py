"""Tests for io module

"""

import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import json as json

import timescaleanalysis
import timescaleanalysis.io

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'


# Test function for io.load_npArray
@pytest.mark.parametrize(
    'input_file, result, error',
    [
        (
            TEST_TRAJ/'test_simple_array.txt',
            np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
            None
        ),
        (
            TEST_TRAJ/'non_existing_file.txt',
            None,
            FileNotFoundError
        ),
    ]
)
def test_load_npArray(
        input_file,
        result,
        error):
    if not error:
        loaded_array = timescaleanalysis.io.load_npArray(input_file)
        np.testing.assert_allclose(loaded_array, result)
    else:
        with pytest.raises(error):
            timescaleanalysis.io.load_npArray(input_file)


# Test function for io.save_npArray
@pytest.mark.parametrize(
    'array, folder_path, file_name, comment', [
        (
            np.array([[1, 2], [3, 4]]),
            TEST_DATA/'test_output_files/',
            'test_data.txt',
            'One test after another'
        ),
    ]
)
def test_save_npArray(
        array,
        folder_path,
        file_name,
        comment):
    timescaleanalysis.io.save_npArray(array, folder_path, file_name, comment)
    assert isfile(folder_path/file_name)
    np.testing.assert_allclose(
        timescaleanalysis.io.load_npArray(folder_path/file_name), array
    )


# Test function for io.save_json
@pytest.mark.parametrize(
        'output_dic, output_path, output_file', [
            (
                {"obs1": [1.0, 1.5, 2.0],
                 "obs2": [1.0, 2.5, 3.0],
                 "obs3": [-0.5, 0.5, -1.5],
                 "labels": ["obs1", "obs2", "obs3"]},
                TEST_DATA/'test_output_files/',
                'test_output_json_file'
            )
        ]
)
def test_save_json(
        output_dic,
        output_path,
        output_file):
    output_file = timescaleanalysis.io.save_json(
        output_dic,
        str(output_path),
        output_file=output_file
    )
    assert isfile(output_file)
    with open(output_file, 'r') as f:
        loaded_dic = json.load(f)
    assert loaded_dic == output_dic
