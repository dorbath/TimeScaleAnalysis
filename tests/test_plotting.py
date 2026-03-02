"""Tests for plotting module

"""

import timescaleanalysis.plotting as plotting

import numpy as np
import pytest
from pathlib import Path
from genericpath import isfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettypyplot as pplt
import json as json
pplt.use_style(colors='cbf8', cmap='macaw_r')

TEST_TRAJ = Path(__file__).parent / 'test_data/test_trajectories'
TEST_DATA = Path(__file__).parent / 'test_data'


# Test function for plotting.plot_TSA
@pytest.mark.parametrize(
        'input_json', [
            TEST_DATA/'test_plotting/test_plot_TSA.json'
        ]
)
def test_plot_TSA(
        input_json):
    with open(input_json) as f:
        input_data = json.load(f)
    ax1, ax2 = plotting.plot_TSA(
        np.asarray(input_data['data_mean']),
        np.asarray(input_data['data_sem']),
        np.asarray(input_data['spectrum']),
        np.asarray(input_data['times']),
        np.asarray(input_data['lag_rates'])
    )
    assert isinstance(ax1, mpl.axes.Axes)
    assert isinstance(ax2, mpl.axes.Axes)

    assert len(ax1.lines) == 2
    assert len(ax2.lines) == 1


# Test function for plotting.plot_dynamical_content
@pytest.mark.parametrize(
        'input_data',
        [
            TEST_DATA/'test_plotting/test_plot_dynCont.txt'
        ]
)
def test_plot_dynamical_content(
        input_data):
    data = np.loadtxt(input_data)
    ax1 = plotting.plot_dynamical_content(data[:, 0], data[:, 1])
    assert isinstance(ax1, mpl.axes.Axes)
    assert len(ax1.lines) == 1


# Test function for plotting.plot_2D_histogram
@pytest.mark.parametrize(
        'xVal, yVal, zVal, error', [
            (
                np.array([0, 1, 2, 3]),
                np.array([0, 1, 2, 3]),
                np.array([[0.1, 0.2, 0.3],
                          [0.2, 0.3, 0.4],
                          [0.3, 0.4, 0.5]]),
                None
            ),
            (
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                np.array([[0.1, 0.2, 0.3],
                          [0.2, 0.3, 0.4],
                          [0.3, 0.4, 0.5]]),
                ValueError
            )
        ]
)
def test_plot_2D_histogram(
        xVal,
        yVal,
        zVal,
        error):
    if not error:
        plotting.plot_2D_histogram(xVal, yVal, zVal)
        assert plt.gca() is not None
        assert len(plt.gca().collections) > 0
    else:
        with pytest.raises(error):
            plotting.plot_2D_histogram(xVal, yVal, zVal)


# Test function for plotting.get_alpha_cmap
@pytest.mark.parametrize(
        'cmap_name, alpha', [
            (
                'macaw_r',
                0.5,
            )
        ]
)
def test_get_alpha_cmap(
        cmap_name,
        alpha):
    alpha_cmap = plotting.get_alpha_cmap(cmap_name, alpha)
    assert isinstance(alpha_cmap, mpl.colors.ListedColormap)


# Test function for plotting.pretty_label
def test_pretty_label():
    pass


# Test function for plotting._log_axis
def test__log_axis():
    pass


# Test function for plotting.save_fig
def test_save_fig():
    pass


# Test function for plotting._color_cycle
def test__color_cycle():
    pass










