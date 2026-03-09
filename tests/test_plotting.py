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
        np.asarray(input_data['times'])
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
            ),
            (
                'viridis',
                0.8,
            )
        ]
)
def test_get_alpha_cmap(
        cmap_name,
        alpha):
    cmap_alpha = plotting.get_alpha_cmap(cmap_name, alpha)
    assert isinstance(cmap_alpha, mpl.colors.ListedColormap)
    ncolors = len(cmap_alpha.colors)
    alpha_n = int(ncolors * alpha)
    np.testing.assert_allclose(
        cmap_alpha.colors[:alpha_n, -1],
        np.linspace(0, 1, alpha_n)[:ncolors]
    )
    np.testing.assert_allclose(
        cmap_alpha.colors[:, :-1],
        plt.get_cmap(cmap_name)(np.arange(plt.get_cmap(cmap_name).N))[:, :-1]
    )


# Test function for plotting.pretty_label
@pytest.mark.parametrize(
        'label, prefix, result', [
            (
                'X_Y',
                'd',
                'd(X,Y)'
            ),
            (
                'x1',
                '',
                'x1'
            )
        ]
)
def test_pretty_label(
        label,
        prefix,
        result):
    new_label = plotting.pretty_label(label, prefix=prefix)
    assert new_label == result


# Test function for plotting._log_axis
@pytest.mark.parametrize(
        'axis, error', [
            (
                'x',
                None
            ),
            (
                'xy',
                None
            ),
            (
                '1',
                ValueError
            )
        ]
)
def test__log_axis(
        axis,
        error):
    fig, ax = plt.subplots()
    if not error:
        plotting._log_axis(ax, axis)
        if axis in ['x', 'xy', 'yx']:
            assert ax.get_xscale() == 'symlog'
        else:
            assert ax.get_xscale() == 'linear'
        if axis in ['y', 'xy', 'yx']:
            assert ax.get_yscale() == 'symlog'
        else:
            assert ax.get_yscale() == 'linear'
    else:
        with pytest.raises(error):
            plotting._log_axis(ax, axis)


# Test function for plotting.save_fig
@pytest.mark.parametrize(
        'path', [
            TEST_DATA/'test_output_files/test_save_fig.pdf'
        ]
)
def test_save_fig(
        path):
    plt.plot()
    plotting.save_fig(str(path))
    assert isfile(path)


# Test function for plotting._color_cycle
@pytest.mark.parametrize(
        'result', [
            np.array([
                '#005B8E',
                '#E69F00',
                '#D55E00',
                '#000000',
                '#BE548F',
                '#009E73',
                '#56B4E9'
            ])
        ]
)
def test__color_cycle(
        result):
    plotting._color_cycle()
    np.testing.assert_equal(
        plt.rcParams['axes.prop_cycle'].by_key()['color'][:7],
        result
    )














