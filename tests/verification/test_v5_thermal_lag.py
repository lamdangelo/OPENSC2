"""V5 — Segregated thermal update: first-order lag of a flushed volume.

A single internal node of volume V receives a constant external inflow
mdot_in at temperature T_in (via the production external_mass_sources +
external_source_temperatures interface) and discharges through a linear
valve to a reservoir. The hydraulic state starts at its exact fixed point
(p_node = p_res + R mdot_in, branch flow = mdot_in), so the pressure problem
is stationary and the test isolates the segregated node-energy update
(update_node_temperatures, backward Euler with inflow-only advection):

    M cp dT/dt = mdot_in cp (T_in - T),  M = rho0 V
    T(t) = T_in + (T0 - T_in) e^{-lambda t},  lambda = mdot_in / M

cp is constant so it cancels; the update reduces exactly to backward Euler
of the scalar lag equation. With beta = 0 the V rho beta dT/dt expansion
source vanishes identically, so the cooling transient must not feed back on
the node pressure at all — asserted to 1e-10 relative.
"""

import numpy as np
import pytest

import analytical
from case_builders import (
    VERIFICATION_FLUID,
    build_network,
    internal,
    reservoir,
    valve,
)
from conductor.conductor_flags import MethodFlag

RESERVOIR_PRESSURE = 1.0e5  # Pa
NODE_VOLUME = 1.0e-3  # m^3 -> M = rho0 V = 1 kg
INFLOW_RATE = 0.01  # kg/s -> lambda = 0.01 1/s, tau = 100 s
INFLOW_TEMPERATURE = 300.0  # K
INITIAL_TEMPERATURE = 350.0  # K
VALVE_RESISTANCE = 1.0e5  # Pa/(kg/s)

FLUID_MASS = VERIFICATION_FLUID.density * NODE_VOLUME
RATE = INFLOW_RATE / FLUID_MASS  # lambda [1/s]
# Exact hydraulic fixed point: the external inflow leaves through the valve,
# R mdot_in = p_node - p_res.
NODE_PRESSURE = RESERVOIR_PRESSURE + VALVE_RESISTANCE * INFLOW_RATE


def lag_network():
    nodes = [
        reservoir("sink", RESERVOIR_PRESSURE, temperature=INITIAL_TEMPERATURE),
        internal(
            "mixer",
            NODE_PRESSURE,
            temperature=INITIAL_TEMPERATURE,
            volume=NODE_VOLUME,
        ),
    ]
    branches = [
        valve(
            "drain",
            "mixer",
            "sink",
            linear_resistance=VALVE_RESISTANCE,
            initial_mass_flow=INFLOW_RATE,
        ),
    ]
    network = build_network(nodes, branches, method=MethodFlag.BACKWARD_EULER)
    network.external_mass_sources["mixer"] = INFLOW_RATE
    network.external_source_temperatures["mixer"] = INFLOW_TEMPERATURE
    return network


def run_lag(network, time_step, number_of_steps):
    """Returns (t, T_mixer, p_mixer, mdot_drain) including the initial state."""
    temperatures = [network.node_temperature_of("mixer")]
    pressures = [network.node_pressure_of("mixer")]
    flows = [network.branch_mass_flow_of("drain")]
    for _ in range(number_of_steps):
        network.step(time_step)
        temperatures.append(network.node_temperature_of("mixer"))
        pressures.append(network.node_pressure_of("mixer"))
        flows.append(network.branch_mass_flow_of("drain"))
    times = time_step * np.arange(number_of_steps + 1)
    return (
        times,
        np.asarray(temperatures),
        np.asarray(pressures),
        np.asarray(flows),
    )


def reference_offset(times):
    return analytical.first_order_lag(
        times, INITIAL_TEMPERATURE, INFLOW_TEMPERATURE, RATE
    ) - INFLOW_TEMPERATURE


def test_first_order_lag_accuracy_and_pressure_invariance(constant_fluid):
    # Backward Euler's relative error on the lag equation is
    # e(t)/(T0-Tin) ~ (lambda dt / 2) (lambda t) e^{-lambda t}, giving a
    # relative L2 error over 5 tau of ~0.36 lambda dt. The spec'd step
    # lambda dt = 0.01 makes that 3.6e-3 -- analytically unreachable for the
    # 1e-3 bound -- so per the agreed plan the accuracy run uses
    # lambda dt = 2.5e-3 (expected error ~9e-4 < 1e-3) and lambda dt = 0.01
    # instead anchors the order study below.
    time_step = 2.5e-3 / RATE
    number_of_steps = int(np.ceil(5.0 / RATE / time_step))
    network = lag_network()
    times, temperatures, pressures, flows = run_lag(
        network, time_step, number_of_steps
    )

    error = analytical.relative_l2_error(
        temperatures - INFLOW_TEMPERATURE, reference_offset(times)
    )
    assert error < 1.0e-3

    # beta = 0: the expansion source V rho beta dT/dt is identically zero and
    # the hydraulic state starts at its exact fixed point, so the pressure
    # must not move beyond linear-solver round-off (1e-10 relative is ~1e-5
    # Pa here, far above round-off, far below any physical feedback).
    assert np.max(np.abs(pressures - NODE_PRESSURE)) < 1.0e-10 * NODE_PRESSURE
    # The discharge flow likewise stays at the fixed point.
    assert np.max(np.abs(flows - INFLOW_RATE)) < 1.0e-12


def test_first_order_lag_order_one(constant_fluid):
    # Step-halving study anchored at the spec'd lambda dt = 0.02..0.0025;
    # the segregated update is exactly backward Euler on the scalar lag
    # equation, so the fitted order must be 1 within 0.2.
    step_sizes = 0.02 / RATE / 2 ** np.arange(4)
    errors = []
    for time_step in step_sizes:
        number_of_steps = int(np.round(5.0 / RATE / time_step))
        network = lag_network()
        times, temperatures, _, _ = run_lag(network, time_step, number_of_steps)
        errors.append(
            analytical.relative_l2_error(
                temperatures - INFLOW_TEMPERATURE, reference_offset(times)
            )
        )
    order = analytical.fitted_order(step_sizes, np.asarray(errors))
    assert order == pytest.approx(1.0, abs=0.2)
