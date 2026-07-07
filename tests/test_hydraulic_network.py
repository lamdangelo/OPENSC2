"""Unit tests of the standalone lumped hydraulic network (stage 0 of the
hydraulic field-circuit coupling).

The solver tests compare against closed-form references:

* quadratic valve between two reservoirs: mdot = sqrt(dp / R_quadratic);
* linear valves in series/parallel: exact resistive-network algebra;
* pump loop: intersection of the pump characteristic with the load curve,
  mdot = sqrt(a0 / (a2 + R_quadratic));
* laminar pipe: Hagen-Poiseuille, mdot = dp * rho * A * D^2 / (32 mu l),
  with the real-helium density and viscosity from the shared CoolProp
  interface;
* node filling: linearized RC transient with the compliance
  C = V * rho * kappa_T of a real-fluid volume, p(t) relaxing exponentially
  to the reservoir pressure with time constant R * C.
"""

import numpy as np
import pytest

import interfaces.coolprop_interface as cpi
from conductor.conductor_flags import MethodFlag
from hydraulics.hydraulic_flags import FluidType
from hydraulics.network import (
    HydraulicNetwork,
    HydraulicNetworkInput,
    NetworkBranchKind,
    NetworkNodeKind,
    PortEnd,
)

HELIUM_TEMPERATURE = 4.5  # K
BLASIUS_FLAG = 110  # smooth-tube laminar + Blasius turbulent, total = max


def reservoir(identifier, pressure, temperature=HELIUM_TEMPERATURE):
    return {
        "identifier": identifier,
        "kind": "reservoir",
        "pressure": pressure,
        "temperature": temperature,
    }


def internal(identifier, pressure, temperature=HELIUM_TEMPERATURE, volume=0.0):
    return {
        "identifier": identifier,
        "kind": "internal",
        "initial_pressure": pressure,
        "initial_temperature": temperature,
        "volume": volume,
    }


def valve(identifier, from_node, to_node, **resistances):
    return {
        "identifier": identifier,
        "kind": "valve",
        "from": from_node,
        "to": to_node,
        **resistances,
    }


def network_mapping(nodes, branches, ports=None):
    mapping = {"fluid_type": "helium", "nodes": nodes, "branches": branches}
    if ports is not None:
        mapping["ports"] = ports
    return mapping


def build_network(nodes, branches, ports=None, method=MethodFlag.BACKWARD_EULER):
    return HydraulicNetwork(
        HydraulicNetworkInput.from_mapping(
            network_mapping(nodes, branches, ports)
        ),
        method=method,
    )


# --------------------------------------------------------------------- #
# Input parsing and validation                                          #
# --------------------------------------------------------------------- #


def test_from_mapping_parses_full_section():
    mapping = network_mapping(
        nodes=[
            reservoir("bath", 3.26e5, temperature=4.1),
            internal("plenum", 3.62e5, temperature=4.1, volume=1.0e-3),
        ],
        branches=[
            {
                "identifier": "circulator",
                "kind": "pump",
                "from": "bath",
                "to": "plenum",
                "characteristic": {
                    "head_at_zero_flow": 4.0e4,
                    "quadratic_coefficient": 1.0e9,
                },
            },
            {
                "identifier": "bypass",
                "kind": "pipe",
                "from": "plenum",
                "to": "bath",
                "length": 5.0,
                "hydraulic_diameter": 1.0e-2,
                "cross_section": 7.85e-5,
                "friction_factor_model": BLASIUS_FLAG,
            },
            valve("relief", "plenum", "bath", quadratic_resistance=1.0e9),
        ],
        ports=[
            {
                "node": "plenum",
                "conductor": "CONDUCTOR_1",
                "channel": "CHAN_1",
                "end": "inlet",
            },
        ],
    )
    inputs = HydraulicNetworkInput.from_mapping(mapping)

    assert inputs.fluid_type is FluidType.HELIUM
    bath, plenum = inputs.nodes
    assert bath.kind is NetworkNodeKind.RESERVOIR
    assert plenum.kind is NetworkNodeKind.INTERNAL
    assert plenum.volume == 1.0e-3
    circulator, bypass, relief = inputs.branches
    assert circulator.kind is NetworkBranchKind.PUMP
    assert circulator.characteristic.head_at_zero_flow == 4.0e4
    assert circulator.characteristic.linear_coefficient == 0.0
    assert circulator.resolved_inertance() == 0.0
    assert bypass.kind is NetworkBranchKind.PIPE
    assert bypass.friction_multiplier == 1.0
    assert bypass.resolved_inertance() == pytest.approx(5.0 / 7.85e-5)
    assert relief.quadratic_resistance == 1.0e9
    (port,) = inputs.ports
    assert port.node == "plenum"
    assert port.end is PortEnd.INLET


@pytest.mark.parametrize(
    "nodes, branches, ports, message",
    [
        (
            [reservoir("a", 1.0e5), reservoir("a", 2.0e5)],
            [],
            None,
            "Duplicate network node",
        ),
        (
            [
                {**reservoir("a", 1.0e5), "volumee": 1.0},
                reservoir("b", 2.0e5),
            ],
            [valve("v", "a", "b", linear_resistance=1.0e9)],
            None,
            "Unknown key",
        ),
        (
            [reservoir("a", 1.0e5)],
            [valve("v", "a", "missing", linear_resistance=1.0e9)],
            None,
            "unknown node",
        ),
        (
            [reservoir("a", 1.0e5), reservoir("b", 2.0e5)],
            [valve("v", "a", "a", linear_resistance=1.0e9)],
            None,
            "different nodes",
        ),
        (
            [internal("a", 1.0e5), internal("b", 2.0e5)],
            [valve("v", "a", "b", linear_resistance=1.0e9)],
            None,
            "no reservoir",
        ),
        (
            [reservoir("a", 1.0e5), reservoir("b", 2.0e5)],
            [valve("v", "a", "b")],
            None,
            "positive linear_resistance",
        ),
        (
            [
                {
                    "identifier": "a",
                    "kind": "internal",
                    "initial_temperature": 4.5,
                }
            ],
            [],
            None,
            "Missing required key",
        ),
        (
            [reservoir("a", 1.0e5), reservoir("b", 2.0e5)],
            [valve("v", "a", "b", linear_resistance=1.0e9)],
            [
                {
                    "node": "missing",
                    "conductor": "CONDUCTOR_1",
                    "channel": "CHAN_1",
                    "end": "inlet",
                }
            ],
            "unknown node",
        ),
        (
            [reservoir("a", 1.0e5), reservoir("b", 2.0e5)],
            [valve("v", "a", "b", linear_resistance=1.0e9)],
            [
                {
                    "node": "a",
                    "conductor": "CONDUCTOR_1",
                    "channel": "CHAN_1",
                    "end": "inlet",
                }
            ],
            "must reference an internal node",
        ),
    ],
)
def test_validation_rejects_bad_input(nodes, branches, ports, message):
    with pytest.raises(ValueError, match=message):
        HydraulicNetworkInput.from_mapping(
            network_mapping(nodes, branches, ports)
        )


def test_unsupported_time_integration_method_rejected():
    inputs = HydraulicNetworkInput.from_mapping(
        network_mapping(
            [reservoir("a", 1.0e5), reservoir("b", 2.0e5)],
            [valve("v", "a", "b", linear_resistance=1.0e9)],
        )
    )
    with pytest.raises(ValueError, match="Unsupported time integration"):
        HydraulicNetwork(inputs, method=MethodFlag.ADAMS_MOULTON_4TH_ORDER)


# --------------------------------------------------------------------- #
# Steady states with analytic references                                 #
# --------------------------------------------------------------------- #


def test_steady_state_quadratic_valve_between_reservoirs():
    pressure_drop = 4.0e4
    quadratic_resistance = 1.0e9
    network = build_network(
        [reservoir("high", 3.6e5), reservoir("low", 3.2e5)],
        [
            valve(
                "v", "high", "low", quadratic_resistance=quadratic_resistance
            ),
            # Same valve oriented against the pressure gradient: the flow
            # must come out negative.
            valve(
                "v_reversed",
                "low",
                "high",
                quadratic_resistance=quadratic_resistance,
            ),
        ],
    )
    network.solve_steady_state()

    expected = np.sqrt(pressure_drop / quadratic_resistance)
    assert network.branch_mass_flow_of("v") == pytest.approx(
        expected, rel=1.0e-6
    )
    assert network.branch_mass_flow_of("v_reversed") == pytest.approx(
        -expected, rel=1.0e-6
    )


def test_steady_state_linear_valve_flow_split():
    p_high, p_low = 6.0e5, 5.0e5
    r_feed, r_1, r_2 = 1.0e9, 2.0e9, 3.0e9
    network = build_network(
        [
            reservoir("high", p_high),
            reservoir("low", p_low),
            internal("junction", 5.5e5),  # volume 0: algebraic KCL row
        ],
        [
            valve("feed", "high", "junction", linear_resistance=r_feed),
            valve("path_1", "junction", "low", linear_resistance=r_1),
            valve("path_2", "junction", "low", linear_resistance=r_2),
        ],
    )
    network.solve_steady_state()

    r_parallel = r_1 * r_2 / (r_1 + r_2)
    total_flow = (p_high - p_low) / (r_feed + r_parallel)
    junction_pressure = p_low + total_flow * r_parallel
    assert network.branch_mass_flow_of("feed") == pytest.approx(
        total_flow, rel=1.0e-8
    )
    assert network.node_pressure_of("junction") == pytest.approx(
        junction_pressure, rel=1.0e-8
    )
    assert network.branch_mass_flow_of("path_1") == pytest.approx(
        (junction_pressure - p_low) / r_1, rel=1.0e-8
    )
    assert network.branch_mass_flow_of("path_2") == pytest.approx(
        (junction_pressure - p_low) / r_2, rel=1.0e-8
    )
    # Exact mass conservation at the junction.
    assert network.branch_mass_flow_of("feed") == pytest.approx(
        network.branch_mass_flow_of("path_1")
        + network.branch_mass_flow_of("path_2"),
        rel=1.0e-12,
    )


def test_steady_state_pump_loop():
    head = 4.0e4
    pump_droop = 1.0e9
    load_resistance = 1.0e9
    reservoir_pressure = 3.2e5
    network = build_network(
        [
            reservoir("bath", reservoir_pressure),
            internal("discharge", 3.4e5),
        ],
        [
            {
                "identifier": "pump",
                "kind": "pump",
                "from": "bath",
                "to": "discharge",
                "characteristic": {
                    "head_at_zero_flow": head,
                    "quadratic_coefficient": pump_droop,
                },
            },
            valve(
                "load",
                "discharge",
                "bath",
                quadratic_resistance=load_resistance,
            ),
        ],
    )
    network.solve_steady_state()

    # Operating point: a0 - a2 mdot^2 = R mdot^2.
    expected_flow = np.sqrt(head / (pump_droop + load_resistance))
    expected_pressure = (
        reservoir_pressure + load_resistance * expected_flow ** 2
    )
    assert network.branch_mass_flow_of("pump") == pytest.approx(
        expected_flow, rel=1.0e-6
    )
    assert network.branch_mass_flow_of("load") == pytest.approx(
        expected_flow, rel=1.0e-6
    )
    assert network.node_pressure_of("discharge") == pytest.approx(
        expected_pressure, rel=1.0e-6
    )


def test_steady_state_laminar_pipe_matches_hagen_poiseuille():
    p_high, p_low = 5.001e5, 5.0e5
    length, diameter = 10.0, 5.0e-4
    area = np.pi / 4.0 * diameter ** 2
    network = build_network(
        [reservoir("high", p_high), reservoir("low", p_low)],
        [
            {
                "identifier": "pipe",
                "kind": "pipe",
                "from": "high",
                "to": "low",
                "length": length,
                "hydraulic_diameter": diameter,
                "cross_section": area,
                "friction_factor_model": BLASIUS_FLAG,
            }
        ],
    )
    network.solve_steady_state()

    properties = cpi.compute_properties(
        FluidType.HELIUM,
        {"rho": "Dmass", "mu": "viscosity"},
        HELIUM_TEMPERATURE,
        0.5 * (p_high + p_low),
    )
    density = float(properties["rho"][0])
    viscosity = float(properties["mu"][0])
    hagen_poiseuille_flow = (
        (p_high - p_low) * density * area * diameter ** 2
        / (32.0 * viscosity * length)
    )
    mass_flow = network.branch_mass_flow_of("pipe")
    reynolds = mass_flow * diameter / (area * viscosity)
    assert reynolds < 1500.0, "test must sit in the laminar regime"
    assert mass_flow == pytest.approx(hagen_poiseuille_flow, rel=5.0e-3)


def test_steady_state_external_mass_source():
    # The stage-1 hook: a port flow entering a node balance shifts the node
    # pressure by q * R through the discharge valve.
    reservoir_pressure = 5.0e5
    resistance = 1.0e8
    source_flow = 1.0e-3
    network = build_network(
        [reservoir("bath", reservoir_pressure), internal("node", 5.0e5)],
        [valve("discharge", "node", "bath", linear_resistance=resistance)],
    )
    network.external_mass_sources["node"] = source_flow
    network.solve_steady_state()

    assert network.branch_mass_flow_of("discharge") == pytest.approx(
        source_flow, rel=1.0e-10
    )
    assert network.node_pressure_of("node") == pytest.approx(
        reservoir_pressure + source_flow * resistance, rel=1.0e-10
    )


# --------------------------------------------------------------------- #
# Transients                                                             #
# --------------------------------------------------------------------- #


def filling_network(method, volume=1.0e-3, pressure_step=5.0e3,
                    resistance=1.0e9):
    """Reservoir -> linear valve -> compliant node, starting below the
    reservoir pressure: a linearized RC circuit with tau = R * C."""
    initial_pressure = 5.0e5
    network = build_network(
        [
            reservoir("bath", initial_pressure + pressure_step),
            internal("vessel", initial_pressure, volume=volume),
        ],
        [valve("fill", "bath", "vessel", linear_resistance=resistance)],
        method=method,
    )
    return network, initial_pressure, pressure_step, resistance


def test_node_capacitance_is_volume_rho_kappa():
    network, initial_pressure, _, _ = filling_network(
        MethodFlag.BACKWARD_EULER
    )
    properties = cpi.compute_properties(
        FluidType.HELIUM,
        {"rho": "Dmass", "kappa": "isothermal_compressibility"},
        HELIUM_TEMPERATURE,
        initial_pressure,
    )
    expected = (
        1.0e-3 * float(properties["rho"][0]) * float(properties["kappa"][0])
    )
    vessel = network.node_index["vessel"]
    assert network.node_capacitance[vessel] == pytest.approx(
        expected, rel=1.0e-12
    )


def test_transient_node_filling_follows_rc_time_constant():
    network, initial_pressure, pressure_step, resistance = filling_network(
        MethodFlag.BACKWARD_EULER
    )
    vessel = network.node_index["vessel"]
    time_constant = resistance * network.node_capacitance[vessel]
    time_step = time_constant / 100.0

    times, pressures = [], []
    for step in range(1, 101):
        network.step(time_step)
        times.append(step * time_step)
        pressures.append(network.node_pressure_of("vessel"))

    analytic = (
        initial_pressure
        + pressure_step * (1.0 - np.exp(-np.array(times) / time_constant))
    )
    deviation = np.abs(np.array(pressures) - analytic) / pressure_step
    assert deviation.max() < 0.03
    assert np.all(np.diff(pressures) > 0.0), "filling must be monotone"


def test_bdf2_is_more_accurate_than_backward_euler():
    errors = {}
    for method in (MethodFlag.BACKWARD_EULER, MethodFlag.BACKWARD_DIFFERENCE_2):
        network, initial_pressure, pressure_step, resistance = (
            filling_network(method, pressure_step=500.0)
        )
        vessel = network.node_index["vessel"]
        time_constant = resistance * network.node_capacitance[vessel]
        time_step = time_constant / 20.0
        worst = 0.0
        for step in range(1, 21):
            network.step(time_step)
            analytic = initial_pressure + pressure_step * (
                1.0 - np.exp(-step * time_step / time_constant)
            )
            worst = max(
                worst,
                abs(network.node_pressure_of("vessel") - analytic)
                / pressure_step,
            )
        errors[method] = worst

    assert (
        errors[MethodFlag.BACKWARD_DIFFERENCE_2]
        < errors[MethodFlag.BACKWARD_EULER]
    )


def test_time_evolution_record():
    network, *_ = filling_network(MethodFlag.BACKWARD_EULER)
    assert network.time_evolution_headers() == [
        "time (s)",
        "pressure_bath (Pa)",
        "pressure_vessel (Pa)",
        "temperature_bath (K)",
        "temperature_vessel (K)",
        "mass_flow_rate_fill (kg/s)",
    ]
    network.record_time_evolution(0.0)
    network.step(0.5)
    network.record_time_evolution(0.5)

    record = network.time_evolution_record
    assert record["time (s)"] == [0.0, 0.5]
    assert record["pressure_vessel (Pa)"][1] == network.node_pressure_of(
        "vessel"
    )
    assert record["mass_flow_rate_fill (kg/s)"][1] == (
        network.branch_mass_flow_of("fill")
    )
    network.clear_time_evolution_record()
    assert all(
        len(values) == 0 for values in network.time_evolution_record.values()
    )


# --------------------------------------------------------------------- #
# Energy transport (network-side node enthalpy balances)                 #
# --------------------------------------------------------------------- #


def test_zero_volume_junction_mixes_inflow_temperatures():
    # Two reservoirs at different temperatures feed a zero-volume junction
    # that drains into a sink: one step after the (isothermal) steady state
    # the junction must sit at the flow-weighted enthalpy mixture of its
    # inflows. The cold feed is declared with reversed orientation
    # (junction -> cold), so its physical inflow carries a negative branch
    # flow and exercises the upwind selection.
    hot_temperature, cold_temperature = 4.8, 4.4
    network = build_network(
        [
            reservoir("hot", 6.0e5, temperature=hot_temperature),
            reservoir("cold", 6.0e5, temperature=cold_temperature),
            reservoir("sink", 5.0e5),
            internal("junction", 5.5e5),
        ],
        [
            valve("feed_hot", "hot", "junction", linear_resistance=1.0e9),
            valve("feed_cold", "junction", "cold", linear_resistance=2.0e9),
            valve("drain", "junction", "sink", linear_resistance=1.0e9),
        ],
    )
    network.solve_steady_state()
    network.step(1.0)

    hot_flow = network.branch_mass_flow_of("feed_hot")
    cold_flow = -network.branch_mass_flow_of("feed_cold")
    assert hot_flow > 0.0 and cold_flow > 0.0
    cp = network.branch_isobaric_specific_heat
    hot_advective = hot_flow * cp[network.branch_index["feed_hot"]]
    cold_advective = cold_flow * cp[network.branch_index["feed_cold"]]
    expected = (
        hot_advective * hot_temperature + cold_advective * cold_temperature
    ) / (hot_advective + cold_advective)
    junction_temperature = network.node_temperature_of("junction")
    assert cold_temperature < junction_temperature < hot_temperature
    assert junction_temperature == pytest.approx(expected, rel=1.0e-12)

    # The mixture is algebraic (no storage): further steps hold it.
    network.step(1.0)
    assert network.node_temperature_of("junction") == pytest.approx(
        junction_temperature, rel=1.0e-9
    )


def test_volume_node_temperature_relaxation():
    # A compliant vessel initialized at the hydraulic fixed point between a
    # hot feed and a cold sink: the flow is steady from the start and the
    # vessel temperature relaxes towards the feed temperature with
    # tau = (rho V cp)_node / (mdot cp_feed). The first step must match the
    # backward Euler update with the frozen coefficients exactly; the full
    # trajectory follows the analytic exponential (small temperature span,
    # so the refrozen properties barely drift).
    hot_temperature, initial_temperature = 4.6, 4.5
    initial_pressure = 5.0e5
    volume = 1.0e-3
    network = build_network(
        [
            reservoir("bath_hot", 5.05e5, temperature=hot_temperature),
            reservoir("sink", 4.95e5),
            internal("vessel", initial_pressure, volume=volume),
        ],
        [
            valve("fill", "bath_hot", "vessel", linear_resistance=1.0e9),
            valve("drain", "vessel", "sink", linear_resistance=1.0e9),
        ],
    )
    network.solve_steady_state()

    vessel = network.node_index["vessel"]
    mass_flow = network.branch_mass_flow_of("fill")
    heat_capacity = (
        volume
        * network.node_density[vessel]
        * network.node_isobaric_specific_heat[vessel]
    )
    advective = mass_flow * network.branch_isobaric_specific_heat[
        network.branch_index["fill"]
    ]
    time_constant = heat_capacity / advective
    time_step = time_constant / 50.0

    network.step(time_step)
    expected_first = (
        heat_capacity / time_step * initial_temperature
        + advective * hot_temperature
    ) / (heat_capacity / time_step + advective)
    assert network.node_temperature_of("vessel") == pytest.approx(
        expected_first, rel=1.0e-12
    )

    temperatures = [network.node_temperature_of("vessel")]
    for _ in range(99):
        network.step(time_step)
        temperatures.append(network.node_temperature_of("vessel"))
    times = time_step * np.arange(1, 101)
    analytic = hot_temperature + (
        initial_temperature - hot_temperature
    ) * np.exp(-times / time_constant)
    deviation = np.abs(np.array(temperatures) - analytic) / (
        hot_temperature - initial_temperature
    )
    assert deviation.max() < 0.05
    assert np.all(np.diff(temperatures) > 0.0), "heating must be monotone"
    # Thermal expansion of the heated vessel acts as a mass source in the
    # node balance and must lift the vessel pressure off the fixed point.
    assert network.node_pressure_of("vessel") > initial_pressure


def test_internal_node_unknown_index():
    network = build_network(
        [reservoir("bath", 5.0e5), internal("vessel", 5.0e5, volume=1.0e-3)],
        [valve("fill", "bath", "vessel", linear_resistance=1.0e9)],
    )
    assert network.internal_node_unknown_index("vessel") == 0
    with pytest.raises(ValueError, match="not an internal node"):
        network.internal_node_unknown_index("bath")


def test_coupled_node_rows_are_backward_euler_under_crank_nicolson():
    # A node flagged as coupled (its balance receives an implicit conductor
    # port flow) must be discretized with backward Euler even when the
    # network runs Crank-Nicolson, matching the fully implicit treatment of
    # the interface on the field side.
    def assembled(method, coupled):
        network, *_ = filling_network(method)
        if coupled:
            network.coupled_node_identifiers.add("vessel")
        network.update_properties()
        return network, *network.assemble_transient(1.0)

    network_cn, matrix_cn, known_cn = assembled(
        MethodFlag.CRANK_NICOLSON, coupled=True
    )
    _, matrix_be, known_be = assembled(MethodFlag.BACKWARD_EULER,
                                       coupled=False)
    _, matrix_cn_plain, _ = assembled(MethodFlag.CRANK_NICOLSON,
                                      coupled=False)
    vessel_row = network_cn.internal_node_unknown_index("vessel")
    assert np.allclose(matrix_cn[vessel_row], matrix_be[vessel_row])
    assert known_cn[vessel_row] == pytest.approx(known_be[vessel_row])
    # Sanity: without the flag the Crank-Nicolson row genuinely differs.
    assert not np.allclose(matrix_cn_plain[vessel_row], matrix_be[vessel_row])


def test_crank_nicolson_enforces_junction_constraint_exactly():
    # A zero-volume junction between two valves is an algebraic constraint
    # row; the implicit enforcement must hold it exactly at every step even
    # under Crank-Nicolson (a plain theta blend would sustain an oscillating
    # constraint residual instead).
    network = build_network(
        [
            reservoir("bath", 5.05e5),
            internal("junction", 5.0e5),
            internal("vessel", 5.0e5, volume=1.0e-3),
        ],
        [
            valve("feed", "bath", "junction", linear_resistance=1.0e9),
            valve("drain", "junction", "vessel", linear_resistance=1.0e9),
        ],
        method=MethodFlag.CRANK_NICOLSON,
    )
    vessel = network.node_index["vessel"]
    time_constant = (
        2.0e9 * network.node_capacitance[vessel]
    )  # series resistance
    time_step = time_constant / 30.0

    for _ in range(30):
        network.step(time_step)
        inflow = network.branch_mass_flow_of("feed")
        outflow = network.branch_mass_flow_of("drain")
        assert abs(inflow - outflow) <= 1.0e-14 + 1.0e-10 * abs(inflow)
    # The vessel must have moved towards the bath pressure.
    assert network.node_pressure_of("vessel") > 5.02e5
