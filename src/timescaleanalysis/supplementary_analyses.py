import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import timescaleanalysis.plotting as plotting


def fit_log_periodic_oscillations(
        log_time_trace: np.array,
        times: np.array,
        fit_range: list,
        popt: tuple = None):
    """Fit the time trace with a power law (t^a0) and
    logarithmic oscillations with period tau_log.
    A system with hierarchical dynamics may yield equidistant timescales/peaks
    in the timescale spectrum (on a log-time axis).
    This will result in logarithmic oscillations superimposed by a power law
    which describes the diffusive dynamics.

    The fit function is

            sa + sb*t^a0 + sc*t^a0 * cos(2pi/tau log10(t) + phi)

    Parameters
    ----------
    log_time_trace: np.array, data points to fit
            (log-spaced and averaged time trace)
    times: np.array, log-spaced time
    fit_range: list, lower and upper boundary of log-fit in [ns]
            (this improves the fit as many time traces converge at some point)
    popt: tuple, initial guess of all 6 parameters:
            (a0, tau, s_a, s_b, s_c, phi)
            a0: exponent of power law
            tau: period of logarithmic oscillations
            s_a: offset of fit function
            s_b: amplitude of power law
            s_c: amplitude of logarithmic oscillations
            phi: phase shift of logarithmic oscillations
    filename: str, name of file to save fit parameters
            (default: None, no saving)
    Return
    ------
    ax1: matplotlib.axes, main plot axis
    ax_insert: matplotlib.axes, inset plot axis
    fitParameters: tuple, optimized fit parameters
            (a0, tau, s_a, s_b, s_c, phi)
            with their standard deviations from the covariance matrix
    """

    def _fit_log_osc(x, a0, tau, sa, sb, sc, phi):
        """Full fit function with power law and logarithmic oscillations"""
        return (sa
                + sb*np.power(x, a0)
                + sc*np.power(x, a0)*np.cos((2*np.pi/tau)*np.log10(x)+phi))

    if any(log_time_trace < 0):
        warnings.warn(
            "Negative values in 'log_time_trace'! "
            "This may lead to problems in the fit and plotting. "
            "It is recommended to shift the time trace to positive values.",
            category=Warning,
        )

    fig, ax1 = plt.subplots()
    ax1.tick_params(direction='in', which='both')

    lower_bound = np.where(times >= fit_range[0])[0][0]
    upper_bound = np.where(times <= fit_range[1])[0][-1]

    # Prevent devision by zero problems
    times[np.isclose(times, 0)] = 1e-10

    fit_times = times[lower_bound:upper_bound]
    fit_time_trace = log_time_trace[lower_bound:upper_bound]
    if popt is None:
        a0 = 0.5  # diffusive process
        tau_log = 2.0  # 1/2 oscillations per decade
        sa = fit_time_trace[0]
        sb = 1.0
        sc = 1.0
        phi = -np.pi/2
        warnings.warn(
            "No initial guess for fit parameters provided! "
            "This may lead to problems in the log oscillation fit. "
            "Initial values are guess as follows: "
            f"a0={a0}, tau_log={tau_log}, "
            f"sa={sa:.4f}, "
            f"sb={sb}, sc={sc}, phi={phi:.4f}.",
            category=Warning,
        )
        popt = (a0, tau_log, sa, sb, sc, phi)

    popt, pcov = curve_fit(
        _fit_log_osc,
        fit_times,
        fit_time_trace,
        p0=popt,
        maxfev=10000
    )

    ax1.plot(times, log_time_trace, label=r'$\langle r(t)\rangle$ [nm]', c='k')
    ax1.plot(fit_times, _fit_log_osc(fit_times, *popt), c='tab:red')
    ax1.grid(False, axis='x')
    plotting._log_axis(ax1, axis='xy')
    plt.tick_params(direction='in', which='both')

    ax_insert = ax1.inset_axes([0.5, 0.05, 0.45, 0.45])
    ax_insert.plot(
        times,
        ((log_time_trace - popt[2]) / np.power(times, popt[0]) - popt[3]),
        color='k'
    )
    ax_insert.plot(
        fit_times,
        (_fit_log_osc(fit_times, *popt)-popt[2])
        / np.power(fit_times, popt[0])-popt[3],
        color='tab:red')
    ax_insert.grid(None)
    ax_insert.set_xlim(ax1.get_xlim()[0], ax1.get_xlim()[1])
    ax_insert.set_ylim(-2.5*popt[4], 2.5*popt[4])
    plotting._log_axis(ax_insert, axis='x')
    ax_insert.yaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.xaxis.set_major_formatter(plt.NullFormatter())
    ax_insert.tick_params(labelsize=0, length=2)
    ax_insert.tick_params(direction='in', which='both', labelsize=8)

    return ax1, ax_insert, np.vstack([popt, np.sqrt(np.diag(pcov))])
