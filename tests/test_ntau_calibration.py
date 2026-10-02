"""Tests for the in-situ n.tau calibration (W7-X quench-test validation).

The calibration script lives in ``W7-X/quench_test_validation`` and is a
standalone module rather than part of the installed package; importing it
prepends both ``source_code`` and its own directory to ``sys.path``.

Covered here (Phase 0.1): the bracket assertion that turns the old silent
``bisection -> high`` failure into a clear error, and the root-enumeration
helpers that let the non-monotone surge metric pick its physical root.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

CALIBRATION_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "W7-X"
    / "quench_test_validation"
)
sys.path.insert(0, str(CALIBRATION_DIRECTORY))

calibration = pytest.importorskip("calibrate_coupling_loss")


def straight_channel(field_tesla: float = 4.0, length: float = 10.0):
    """One synthetic channel with a uniform field, as (x, B)."""
    x = np.linspace(0.0, length, 50)
    return [(x, np.full_like(x, field_tesla))]


def test_roots_on_grid_finds_both_crossings_of_a_non_monotone_metric():
    grid = np.logspace(-3, 1, 200)
    # A concave-down curve in log(n.tau): rises, peaks, falls -> two roots.
    values = -((np.log10(grid) - 0.0) ** 2) + 1.0
    roots = calibration.roots_on_grid(values, 0.0, grid)
    assert len(roots) == 2
    assert roots[0] < 1.0 < roots[1]
    np.testing.assert_allclose(roots, [0.1, 10.0], rtol=1e-2)


def test_refine_root_converges_inside_a_sign_changing_bracket():
    root = calibration.refine_root(lambda x: x**2 - 2.0, 1.0, 2.0)
    assert abs(root - np.sqrt(2.0)) < 1e-6


def test_solve_ntau_returns_root_when_target_is_bracketed():
    inverter = calibration.AdiabaticTemperature(4.5, include_glass_epoxy=True)
    channels = straight_channel()
    rate_integral = 1.0e8
    low, high = calibration.NTAU_BRACKET
    metric = "outlet end"
    f_low = calibration.predicted_outlet_temperatures(
        low, channels, inverter, rate_integral
    )[metric]
    f_high = calibration.predicted_outlet_temperatures(
        high, channels, inverter, rate_integral
    )[metric]
    target = 0.5 * (f_low + f_high)
    ntau = calibration.solve_ntau(target, metric, channels, inverter, rate_integral)
    assert low < ntau < high
    recovered = calibration.predicted_outlet_temperatures(
        ntau, channels, inverter, rate_integral
    )[metric]
    assert abs(recovered - target) < 1e-3


def test_solve_ntau_raises_when_target_exceeds_bracket():
    """Target above f(high) must error, not silently return ~high."""
    inverter = calibration.AdiabaticTemperature(4.5, include_glass_epoxy=True)
    channels = straight_channel()
    rate_integral = 1.0e8
    high = calibration.NTAU_BRACKET[1]
    f_high = calibration.predicted_outlet_temperatures(
        high, channels, inverter, rate_integral
    )["outlet end"]
    with pytest.raises(ValueError, match="does not enclose"):
        calibration.solve_ntau(
            f_high + 10.0, "outlet end", channels, inverter, rate_integral
        )


# --- Phase 1.1: closed-form expansion outflow -----------------------------


def synthetic_template():
    """A smooth decay template (dI/dt)^2 with a positive cumulative S(t)."""
    time = np.linspace(0.0, 4.0, 800)
    rate = -8000.0 * np.exp(-time / 1.5)  # dI/dt of an exponential decay
    return time, rate**2


def euler_reference_outflow(model, ntau, channels, template, fit_times, time_step):
    """Fixed-step forward-Euler expansion outflow (test-only reference).

    An independent integrator of the equilibrated balance
    M'(T) dT/dt = q'(x,t): it exercises the same shape(x), M' and expulsion
    quadrature as the production closed form but reaches the temperature by
    time stepping instead of the analytic inverse, so its dt->0 agreement
    is a wiring check the ``n.tau*S`` identity cannot give. Time is indexed
    by integer step (never a rounded float key).
    """
    template_time, squared_rate = template
    step_of_fit = {ft: int(round(ft / time_step)) for ft in fit_times}
    results = {ft: 0.0 for ft in fit_times}
    for x, field in channels:
        shape = model._channel_shape(ntau, field)
        temperature = np.full_like(field, model._initial_temperature)
        history = {}
        for step in range(max(step_of_fit.values()) + 1):
            power = shape * np.interp(step * time_step, template_time, squared_rate)
            expelled = (
                np.interp(temperature, model._grid, model._expulsion_factor) * power
            )
            history[step] = float(np.trapezoid(expelled, x))
            temperature = temperature + time_step * power * np.interp(
                temperature, model._grid, model._inverse_capacity
            )
        for fit_time in fit_times:
            results[fit_time] += history[step_of_fit[fit_time]]
    return results


def test_closed_form_matches_fine_step_euler():
    """The closed form is the dt->0 limit of the forward-Euler integrator."""
    model = calibration.HeliumExpansionModel(6.248)
    channels = straight_channel(field_tesla=3.5, length=12.0)
    template = synthetic_template()
    fit_times = (2.0, 3.0)
    closed = model.excess_outflow(0.05, channels, template, fit_times)
    fine = euler_reference_outflow(
        model, 0.05, channels, template, fit_times, time_step=0.001
    )
    for fit_time in fit_times:
        assert closed[fit_time] > 0.0
        assert abs(fine[fit_time] / closed[fit_time] - 1.0) < 2e-3


def test_euler_converges_to_closed_form_under_step_halving():
    model = calibration.HeliumExpansionModel(6.248)
    channels = straight_channel(field_tesla=3.5, length=12.0)
    template = synthetic_template()
    closed = model.excess_outflow(0.05, channels, template, (2.5,))[2.5]
    errors = [
        abs(
            euler_reference_outflow(
                model, 0.05, channels, template, (2.5,), time_step=dt
            )[2.5]
            / closed
            - 1.0
        )
        for dt in (0.04, 0.02, 0.01)
    ]
    # Monotone reduction of the Euler error as the step is halved.
    assert errors[0] >= errors[1] >= errors[2]


def test_temperature_depends_only_on_ntau_times_S():
    """T(x,t) enters through the product n.tau * S(t) alone (1.1 identity)."""
    model = calibration.HeliumExpansionModel(6.248)
    x, field = straight_channel(field_tesla=4.0)[0]
    time, cumulative = calibration.cumulative_rate_integral(synthetic_template())

    def temperature(ntau, fit_time):
        energy = model._channel_shape(ntau, field) * float(
            np.interp(fit_time, time, cumulative)
        )
        return model._inverter(energy)

    ntau1, t1 = 0.05, 2.5
    s1 = float(np.interp(t1, time, cumulative))
    ntau2 = 0.075
    t2 = float(np.interp(ntau1 * s1 / ntau2, cumulative, time))  # n.tau*S equal
    np.testing.assert_allclose(temperature(ntau1, t1), temperature(ntau2, t2),
                               atol=1e-9)


# --- Phase 2.1: 2-node systematic and the AAB25 band ----------------------


def test_two_node_reduces_to_closed_form_as_h_to_infinity():
    """The 2-node model's h->inf limit is the equilibrated closed form."""
    template = synthetic_template()
    fit_times = (1.0, 2.0, 3.0)
    ratio_inf = calibration.two_node_outflow_ratio(6.3, 1.0e8, template, fit_times)
    np.testing.assert_allclose(ratio_inf, 1.0, atol=2e-3)
    ratio_finite = calibration.two_node_outflow_ratio(6.3, 100.0, template, fit_times)
    assert np.any(np.abs(ratio_finite - 1.0) > 1e-3)  # finite h departs


def test_aab25_trend_band_dominates_noise_band():
    """The trend band (adds |excess|) is never below the noise-only band."""
    for group in (1, 2, 3, 4):
        trend, _ = calibration.aab25_outflow_band(group, band_mode="trend")
        noise, _ = calibration.aab25_outflow_band(group, band_mode="noise")
        for fit_time in calibration.AAB25_FIT_TIMES:
            assert trend[fit_time] >= noise[fit_time] - 1e-18
            assert trend[fit_time] > 0.0
