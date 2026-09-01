"""V2 — Hydraulic series RLC: surge-tank mass oscillation.

Reservoir -> laminar pipe (inertance L = length/A, Hagen-Poiseuille
resistance R) -> dead-ended internal node with volume V (hydraulic
capacitance C = V rho0 kappa_T). Initial condition: node pressure offset
dp0 above the reservoir, zero flow. The transient network solve (production
path HydraulicNetwork.step -> _assemble_raw -> assemble_transient -> advance,
strictly fixed dt because step() never adapts the caller-supplied time step)
is compared against the closed-form underdamped free response

    p(t) - p_res = dp0 e^{-sigma t} [cos(omega_d t)
                                     + (sigma/omega_d) sin(omega_d t)]
    sigma = R/(2L),  omega_d = sqrt(1/(LC) - sigma^2)

(Wylie & Streeter, "Fluid Transients in Systems", surge-tank oscillation
with linearized friction). Parameters give sigma/omega_0 = 0.05 (underdamped,
>= 8 clean periods) with peak Reynolds ~300, well inside the exactly-laminar
friction band. Both MNA rows carry mass (node volume > 0, pipe inertance
> 0), so Crank-Nicolson and Galerkin are genuine theta schemes here (the
solver's forced-backward-Euler treatment of zero-mass rows touches nothing).
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

import analytical
from case_builders import (
    VERIFICATION_FLUID,
    build_network,
    internal,
    pipe,
    reservoir,
    valve,
)
from conductor.conductor_flags import MethodFlag

RESERVOIR_PRESSURE = 1.0e5  # Pa
PRESSURE_OFFSET = 1.0e3  # Pa (dp0)
PIPE_LENGTH = 10.0  # m
DIAMETER = 1.0e-2  # m
CROSS_SECTION = np.pi / 4.0 * DIAMETER**2  # m^2
DAMPING_RATIO = 0.05  # sigma / omega_0

RESISTANCE = analytical.laminar_pipe_resistance(
    VERIFICATION_FLUID.viscosity,
    PIPE_LENGTH,
    DIAMETER,
    VERIFICATION_FLUID.density,
    CROSS_SECTION,
)
INERTANCE = analytical.pipe_inertance(PIPE_LENGTH, CROSS_SECTION)
SIGMA = RESISTANCE / (2.0 * INERTANCE)
OMEGA_0 = SIGMA / DAMPING_RATIO
# Node volume chosen to hit omega_0 exactly: C = 1/(omega_0^2 L),
# V = C/(rho kappa_T).
CAPACITANCE = 1.0 / (OMEGA_0**2 * INERTANCE)
NODE_VOLUME = CAPACITANCE / (
    VERIFICATION_FLUID.density * VERIFICATION_FLUID.isothermal_compressibility
)
OMEGA_D = np.sqrt(OMEGA_0**2 - SIGMA**2)
DAMPED_PERIOD = 2.0 * np.pi / OMEGA_D


def rlc_network(method):
    nodes = [
        reservoir("supply", RESERVOIR_PRESSURE),
        internal("tank", RESERVOIR_PRESSURE + PRESSURE_OFFSET, volume=NODE_VOLUME),
    ]
    branches = [
        pipe("line", "supply", "tank", PIPE_LENGTH, DIAMETER, CROSS_SECTION),
    ]
    return build_network(nodes, branches, method=method)


def run_free_response(network, time_step, number_of_steps):
    """Fixed-dt transient via the production step(); returns (t, p_tank)
    including the initial state at t = 0."""
    pressures = [network.node_pressure_of("tank")]
    for _ in range(number_of_steps):
        network.step(time_step)
        pressures.append(network.node_pressure_of("tank"))
    times = time_step * np.arange(number_of_steps + 1)
    return times, np.asarray(pressures)


def analytical_offset(times):
    return analytical.rlc_free_response(times, PRESSURE_OFFSET, SIGMA, OMEGA_D)


def test_bdf2_free_response_accuracy(constant_fluid):
    # omega_d dt = 0.01 over 8 damped periods (~5030 steps). BDF2 is globally
    # second order; the dominant error is the O((omega dt)^2) phase drift
    # accumulated over Phi = 16 pi radians, of order (omega dt)^2 Phi/6 ~
    # 8e-4 relative — the 1e-3 bound checks second-order accuracy with the
    # correct constant, and any first-order pollution (e.g. from property
    # freezing) would blow it by two orders of magnitude.
    time_step = 0.01 / OMEGA_D
    number_of_steps = int(np.ceil(8.0 * DAMPED_PERIOD / time_step))
    network = rlc_network(MethodFlag.BACKWARD_DIFFERENCE_2)
    times, pressures = run_free_response(network, time_step, number_of_steps)

    error = analytical.relative_l2_error(
        pressures - RESERVOIR_PRESSURE, analytical_offset(times)
    )
    assert error < 1.0e-3


@pytest.mark.parametrize(
    "method, expected_order",
    [
        (MethodFlag.BACKWARD_EULER, 1.0),
        (MethodFlag.CRANK_NICOLSON, 2.0),
        # Galerkin theta = 2/3 != 1/2, so its truncation error is first
        # order despite belonging to the same theta family as CN.
        (MethodFlag.GALERKIN, 1.0),
        (MethodFlag.BACKWARD_DIFFERENCE_2, 2.0),
    ],
)
def test_time_integrator_convergence_order(constant_fluid, method, expected_order):
    # Fixed time stepping is guaranteed structurally: HydraulicNetwork.step
    # integrates exactly one caller-supplied dt and contains no adaptivity,
    # so repeating step(dt) yields a uniform grid. Errors over 2 damped
    # periods at dt0 = T_d/100 halved three times (coarsest omega_d dt =
    # 0.063: coarse enough that the finest error stays far above round-off,
    # fine enough that the first-order methods are asymptotic — at T_d/25
    # backward Euler's error saturates near the signal norm and the fitted
    # slope collapses to 0.8). The fitted slope must match the theoretical
    # order within 0.2, tight enough to separate order 1 from order 2.
    base_step = DAMPED_PERIOD / 100.0
    step_sizes = base_step / 2 ** np.arange(4)
    errors = []
    for time_step in step_sizes:
        number_of_steps = int(np.round(2.0 * DAMPED_PERIOD / time_step))
        network = rlc_network(method)
        times, pressures = run_free_response(network, time_step, number_of_steps)
        errors.append(
            analytical.relative_l2_error(
                pressures - RESERVOIR_PRESSURE, analytical_offset(times)
            )
        )
    order = analytical.fitted_order(step_sizes, np.asarray(errors))
    assert order == pytest.approx(expected_order, abs=0.2)


def test_fitted_decay_rate_and_frequency(constant_fluid):
    # Extract sigma and omega_d from the finest BDF2 trace by fitting the
    # exp-cosine free response. 0.5% bounds the combination of the BDF2
    # discretization bias at omega_d dt = 0.01 (numerical frequency shift
    # ~ (omega dt)^2/6 ~ 2e-5 relative, decay-rate bias of the same order
    # relative to omega_0, i.e. ~4e-4 relative to sigma) and the fit's
    # insensitivity floor; a wrong R, L or C by >1% would violate it.
    time_step = 0.01 / OMEGA_D
    number_of_steps = int(np.ceil(8.0 * DAMPED_PERIOD / time_step))
    network = rlc_network(MethodFlag.BACKWARD_DIFFERENCE_2)
    times, pressures = run_free_response(network, time_step, number_of_steps)

    def model(time, amplitude, sigma, omega_d):
        return amplitude * np.exp(-sigma * time) * (
            np.cos(omega_d * time) + (sigma / omega_d) * np.sin(omega_d * time)
        )

    fitted, _ = curve_fit(
        model,
        times,
        pressures - RESERVOIR_PRESSURE,
        p0=[PRESSURE_OFFSET, SIGMA, OMEGA_D],
    )
    _, fitted_sigma, fitted_omega_d = fitted
    assert fitted_sigma == pytest.approx(SIGMA, rel=5.0e-3)
    assert fitted_omega_d == pytest.approx(OMEGA_D, rel=5.0e-3)


def test_quadratic_valve_plausibility(constant_fluid):
    # QUALITATIVE check only (clearly marked per the suite ground rules):
    # with a quadratic valve replacing the pipe the damping is amplitude-
    # dependent (the solver applies a tangent linearization frozen at the old
    # flow), so no closed-form trace exists. Assert only that the response
    # remains an oscillatory decay at the undamped frequency: monotonically
    # decreasing envelope and mean period within 5% of 2 pi / omega_0.
    #
    # K sized by equivalent linearization (8/(3 pi)) K |mdot_peak| = R so the
    # initial decay matches the linear case's scale.
    peak_flow = PRESSURE_OFFSET * np.sqrt(CAPACITANCE / INERTANCE)
    quadratic_resistance = RESISTANCE * 3.0 * np.pi / (8.0 * peak_flow)
    nodes = [
        reservoir("supply", RESERVOIR_PRESSURE),
        internal("tank", RESERVOIR_PRESSURE + PRESSURE_OFFSET, volume=NODE_VOLUME),
    ]
    branches = [
        valve(
            "orifice",
            "supply",
            "tank",
            quadratic_resistance=quadratic_resistance,
            inertance=INERTANCE,
        ),
    ]
    network = build_network(nodes, branches, method=MethodFlag.BACKWARD_DIFFERENCE_2)

    time_step = DAMPED_PERIOD / 200.0
    number_of_steps = int(np.ceil(8.0 * DAMPED_PERIOD / time_step))
    times, pressures = run_free_response(network, time_step, number_of_steps)
    offset = pressures - RESERVOIR_PRESSURE

    # Oscillatory: several sign changes.
    sign_changes = np.sum(np.abs(np.diff(np.sign(offset))) > 0)
    assert sign_changes >= 5

    # Envelope decays monotonically: successive local-maximum magnitudes of
    # |offset| strictly decrease.
    interior = np.arange(1, offset.size - 1)
    peaks = interior[
        (np.abs(offset[interior]) >= np.abs(offset[interior - 1]))
        & (np.abs(offset[interior]) > np.abs(offset[interior + 1]))
    ]
    peak_values = np.abs(offset[peaks])
    assert peak_values.size >= 6
    assert np.all(np.diff(peak_values) < 0.0)

    # Period within 5% of the undamped 2 pi / omega_0: successive
    # same-direction zero crossings are one period apart.
    crossing_indices = np.nonzero(np.diff(np.sign(offset)) < 0)[0]
    crossing_times = times[crossing_indices]
    measured_period = np.mean(np.diff(crossing_times))
    assert measured_period == pytest.approx(2.0 * np.pi / OMEGA_0, rel=0.05)
