"""Unit tests of the Tier-2 transmission-line reference (no main-solver
import): lossless d'Alembert propagation and Joukowsky amplitude, closed /
open terminations (Gamma = +1 / -1), friction damping rate, and the grid
convergence order of the scheme."""

import numpy as np
import pytest

from reference.transmission_line import (
    NodeTermination,
    propagation_constant,
    reflection_coefficient_volume,
    sinusoid,
    smooth_step,
    solve_transmission_line,
    uniform_line,
)

RHO, C, AREA, LENGTH = 1000.0, 30.0, 1.0e-4, 10.0
Z = C / AREA  # Pa/(kg/s)
STEP = 100.0  # Pa
RAMP = 0.04  # s (1.2 m front)


def pinned_supply(ramp=RAMP, schedule=None):
    """Supply node clamped to the reservoir: algebraic node (C = 0) with a
    negligible far-side resistance."""
    return NodeTermination(capacitance=0.0, far_resistance=1e-3 * Z,
                           reservoir=schedule or smooth_step(STEP, ramp))


def run(points, return_node, t_end, snapshot_times=(), cfl=0.1):
    line = uniform_line("line", LENGTH, points, AREA, RHO, C)
    return solve_transmission_line([line], pinned_supply(), return_node, t_end,
                                   snapshot_times, cfl=cfl, series_stride=1)


def test_joukowsky_front_and_transit_time():
    """Before the front reaches the far end the line is semi-infinite: the
    port mass flow equals dp/Z and the front arrives at L/c."""
    transit = LENGTH / C
    solution = run(401, NodeTermination(0.0, 1e12 * Z), 0.9 * transit,
                   snapshot_times=[0.5 * transit])
    flow = solution.port_flow_inlet["line"]
    times = solution.times
    late = times > 2 * RAMP
    assert np.allclose(flow[late], STEP / Z, rtol=2e-3)
    pressure = solution.snapshots["line"]["pressure"][0]
    x = solution.snapshots["line"]["x"]
    # The half-amplitude point of the (symmetric) smooth step sits at
    # c (t - RAMP/2).
    front = x[np.argmax(pressure < 0.5 * STEP)]
    assert front == pytest.approx(0.5 * transit * C - 0.5 * RAMP * C, abs=0.05)


@pytest.mark.parametrize(
    "far_resistance, expected",
    [(1e6 * Z, +1.0), (1e-6 * Z, -1.0)],
)
def test_closed_and_open_terminations(far_resistance, expected):
    """Closed end (R -> inf) doubles the pressure step, open end (R -> 0)
    cancels it: reflection coefficient +1 / -1 measured from the plateau at
    the mid-point after the first reflection."""
    transit = LENGTH / C
    # At 1.7 transit times the reflected front (1.2 m wide) sits at 0.3 L,
    # clear of the measurement window around the mid-point.
    solution = run(801, NodeTermination(0.0, far_resistance), 1.8 * transit,
                   snapshot_times=[1.7 * transit])
    pressure = solution.snapshots["line"]["pressure"][0]
    x = solution.snapshots["line"]["x"]
    middle = np.abs(x - 0.5 * LENGTH) < 0.05 * LENGTH
    plateau = pressure[middle].mean()
    gamma = plateau / STEP - 1.0
    assert gamma == pytest.approx(expected, abs=3e-3)


def test_friction_attenuation_and_phase():
    """Sinusoidal drive on a long lossy line before any reflection returns:
    the amplitude ratio and phase lag between two probes equal exp(-alpha d)
    and beta d of the telegrapher propagation constant."""
    F = 2.0  # 1/s
    omega = 2 * np.pi * 10.0  # rad/s, wavelength 3 m
    line = uniform_line("line", 3 * LENGTH, 1201, AREA, RHO, C, friction_tangent=F)
    probes = (0.5 * LENGTH, LENGTH)
    t_end = 0.8 * LENGTH / C * 2.5  # 0.67 s: reflection from x = 3L reaches
    # x = L only at (3L + 2L)/c = 1.67 s
    solution = solve_transmission_line(
        [line], pinned_supply(schedule=sinusoid(STEP, omega, 0.1)),
        NodeTermination(0.0, 1e12 * Z), t_end,
        np.linspace(t_end - 2 * np.pi / omega, t_end, 41),
    )
    x = solution.snapshots["line"]["x"]
    times = solution.snapshot_times
    traces = [solution.snapshots["line"]["pressure"][:, np.argmin(np.abs(x - probe))]
              for probe in probes]
    # Least-squares fit of a sinusoid at omega over the last period.
    basis = np.column_stack([np.sin(omega * times), np.cos(omega * times)])
    phasors = []
    for trace in traces:
        coefficients, *_ = np.linalg.lstsq(basis, trace, rcond=None)
        # A sin(wt) + B cos(wt) = Im[(A + iB) e^{i w t}]
        phasors.append(coefficients[0] + 1j * coefficients[1])
    ratio = phasors[1] / phasors[0]
    gamma = propagation_constant(omega, C, F)
    distance = probes[1] - probes[0]
    assert abs(ratio) == pytest.approx(np.exp(-gamma.real * distance), rel=2e-3)
    # Phase lag beta d (mod 2 pi); the fitted phasor is exp(-i beta d).
    lag = -np.angle(ratio)
    expected_lag = (gamma.imag * distance) % (2 * np.pi)
    assert lag % (2 * np.pi) == pytest.approx(expected_lag, abs=2e-3)


def test_grid_convergence_order():
    """Self-convergence of the mid-line pressure profile at fixed CFL: the
    scheme is 4th order in space (RK4 error at CFL 0.1 is far below)."""
    transit = LENGTH / C
    profiles = {}
    grids = (201, 401, 801, 1601)
    for points in grids:
        solution = run(points, NodeTermination(0.0, Z), 0.8 * transit,
                       snapshot_times=[0.8 * transit])
        profiles[points] = (solution.snapshots["line"]["x"],
                            solution.snapshots["line"]["pressure"][0])
    x_fine, p_fine = profiles[grids[-1]]
    errors = []
    for points in grids[:-1]:
        x, p = profiles[points]
        errors.append(np.sqrt(np.mean((p - np.interp(x, x_fine, p_fine)) ** 2)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    # Pre-asymptotic on the coarsest pair (the 1.2 m front is ~8 cells wide
    # at 201 points); the asymptotic order of the 4th-order interior with the
    # cubic boundary closure is between 3 and 4.
    assert orders[-1] > 3.0, orders
    assert np.all(np.diff(orders) > 0.0), orders


def test_analytic_reflection_limits():
    assert reflection_coefficient_volume(Z, 1e12, 0.0, 1.0) == pytest.approx(1.0)
    assert reflection_coefficient_volume(Z, 1e-12, 0.0, 1.0) == pytest.approx(-1.0)
    assert reflection_coefficient_volume(Z, Z, 0.0, 1.0) == pytest.approx(0.0)
