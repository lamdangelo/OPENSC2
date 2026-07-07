"""Stage-1 acceptance test of the hydraulic field-circuit coupling.

The degenerate limit of the port coupling must reproduce the fixed-pressure
boundary condition it replaces: CASE_1 is run twice on a shortened scenario
(2 s, 50 elements), once as committed (CHAN_1 with imposed inlet velocity
and outlet pressure) and once with the CHAN_1 outlet coupled to a minimal
network -- a zero-volume manifold node discharging through an almost ideal
valve into a reservoir at exactly the imposed outlet pressure. The network
then pins the port pressure to p_reservoir + R * mdot, so every field of
every component must match the baseline to solver roundoff. This exercises
the whole coupled path: input parsing, port resolution, the
boundary-condition overlay, the bordered Schur solve and the network state
advance.

The valve resistance must be extremely small: the one-step linear response
of the discretized channel to an outlet-pressure boundary perturbation is
strongly amplified (measured on the baseline operator: ~3e3 /Pa on the
interior pressure and ~3e2 K/Pa on the outlet temperature, through the
pressure-work coupling of the outlet energy row), so R = 1e-9 Pa s/kg keeps
the R * mdot offset (~1e-11 Pa) below double-precision resolution of the
590 kPa boundary value and the residual deviation at pure roundoff level.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulation import Simulation

from regression_utilities import copy_input_files

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_1_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "TDD_examples" / "CASE_1_ITER_like_LTS"
)

SHORTENED_END_TIME = 2.0
NUMBER_OF_ELEMENTS = 50
OUTLET_PRESSURE = 590000.0  # CHAN_1/CHAN_2 outlet_pressure of CASE_1
VALVE_RESISTANCE = 1.0e-9  # Pa/(kg/s): almost ideal discharge valve

NETWORK_SECTION = {
    "fluid_type": "helium",
    "nodes": [
        {
            "identifier": "outlet_manifold",
            "kind": "internal",
            "initial_pressure": OUTLET_PRESSURE,
            "initial_temperature": 4.5,
        },
        {
            "identifier": "recovery_bath",
            "kind": "reservoir",
            "pressure": OUTLET_PRESSURE,
            "temperature": 4.5,
        },
    ],
    "branches": [
        {
            "identifier": "recovery_line",
            "kind": "valve",
            "from": "outlet_manifold",
            "to": "recovery_bath",
            "linear_resistance": VALVE_RESISTANCE,
        },
    ],
    "ports": [
        {
            "node": "outlet_manifold",
            "conductor": "CONDUCTOR_1",
            "channel": "CHAN_1",
            "end": "outlet",
        },
    ],
}


def prepare_run_directory(tmp_path, name: str, network_section=None,
                          end_time: float = SHORTENED_END_TIME,
                          conductor_edits=None) -> Path:
    """Copy the CASE_1 YAML inputs and edit them into the shortened
    scenario: optionally add a hydraulic network section and apply extra
    in-place edits to the conductor document (callback)."""
    run_directory = tmp_path / name
    run_directory.mkdir()
    copy_input_files(CASE_1_INPUT_DIRECTORY, run_directory, "yaml")

    simulation_file = run_directory / "simulation.yaml"
    simulation_document = yaml.safe_load(simulation_file.read_text())
    simulation_document["simulation"]["end_time"] = end_time
    if network_section is not None:
        simulation_document["hydraulic_network"] = copy.deepcopy(
            network_section
        )
    simulation_file.write_text(
        yaml.safe_dump(simulation_document, sort_keys=False)
    )

    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    conductor_document = yaml.safe_load(conductor_file.read_text())
    conductor_document["grid"]["number_of_elements"] = NUMBER_OF_ELEMENTS
    diagnostics = conductor_document["diagnostics"]
    diagnostics["spatial_distribution_times"] = [
        save_time
        for save_time in diagnostics["spatial_distribution_times"]
        if save_time <= end_time
    ]
    if conductor_edits is not None:
        conductor_edits(conductor_document)
    conductor_file.write_text(
        yaml.safe_dump(conductor_document, sort_keys=False)
    )
    return run_directory


def impose_pressure_drop_on_both_channels(conductor_document) -> None:
    """Switch CHAN_1 and CHAN_2 to INTIAL = 1 (imposed pressure drop).

    The two channels form one hydraulic-parallel group (open interface),
    whose initialization requires a uniform INTIAL; the pressure values
    (600 kPa / 590 kPa) are already in the committed input."""
    for component in conductor_document["components"]:
        if component["kind"] == "CHAN":
            component["operations"]["hydraulic_boundary_condition"] = 1


def run_simulation(run_directory: Path) -> Simulation:
    simulation = Simulation(str(run_directory))
    simulation.run()
    return simulation


# The two runs solve the same matrices but along different arithmetic paths
# (single vs stacked right-hand sides plus the border correction), and the
# channel operator amplifies roundoff-level boundary differences (see module
# docstring); 1e-6 keeps a ~6x margin over the observed roundoff floor
# (~1.6e-7) while genuine coupling defects showed up at 1e-5 and above.
FIELD_MATCH_TOLERANCE = 1.0e-6


def assert_fields_match(reference: np.ndarray, coupled: np.ndarray,
                        label: str,
                        tolerance: float = FIELD_MATCH_TOLERANCE) -> None:
    scale = np.abs(reference).max()
    deviation = np.abs(coupled - reference).max() / scale
    assert deviation < tolerance, (
        f"{label}: relative deviation {deviation:.3e} exceeds {tolerance:.0e}"
    )


def assert_conductors_match(baseline: Simulation, coupled: Simulation,
                            conductor_index: int = 0):
    baseline_conductor = baseline.list_of_Conductors[conductor_index]
    coupled_conductor = coupled.list_of_Conductors[conductor_index]

    for base_comp, coup_comp in zip(
        baseline_conductor.inventory.fluids.collection,
        coupled_conductor.inventory.fluids.collection,
    ):
        assert base_comp.identifier == coup_comp.identifier
        for field_name in ("velocity", "pressure", "temperature"):
            assert_fields_match(
                getattr(base_comp.coolant.node_fields, field_name),
                getattr(coup_comp.coolant.node_fields, field_name),
                f"{base_comp.identifier} {field_name}",
            )
    for base_comp, coup_comp in zip(
        baseline_conductor.inventory.solids.collection,
        coupled_conductor.inventory.solids.collection,
    ):
        assert_fields_match(
            base_comp.node_fields.temperature,
            coup_comp.node_fields.temperature,
            f"{base_comp.identifier} temperature",
        )


def test_degenerate_network_matches_fixed_pressure_boundary(tmp_path):
    baseline = run_simulation(
        prepare_run_directory(tmp_path, "baseline")
    )
    coupled = run_simulation(
        prepare_run_directory(
            tmp_path, "coupled", network_section=NETWORK_SECTION
        )
    )

    assert_conductors_match(baseline, coupled)
    coupled_conductor = coupled.list_of_Conductors[0]

    # Network-side consistency: the manifold sits R * mdot above the
    # reservoir, and the recovery line carries the CHAN_1 outlet flow (the
    # zero-volume node balance ties them; densities are frozen one step
    # back, hence the modest tolerance against the freshly recomputed
    # outlet flow).
    network = coupled.hydraulic_network
    line_flow = network.branch_mass_flow_of("recovery_line")
    chan_1 = next(
        f_comp
        for f_comp in coupled_conductor.inventory.fluids.collection
        if f_comp.identifier == "CHAN_1"
    )
    outlet_flow = chan_1.coolant.node_fields.mass_flow_rate[-1]
    np.testing.assert_allclose(line_flow, outlet_flow, rtol=1.0e-3)
    np.testing.assert_allclose(
        network.node_pressure_of("outlet_manifold"),
        OUTLET_PRESSURE + VALVE_RESISTANCE * line_flow,
        rtol=1.0e-12,
    )

    # Network-side energy transport in the degenerate limit: the manifold
    # is a zero-volume node with a single (port) inflow, so its algebraic
    # enthalpy balance pins it to the CHAN_1 outlet temperature, while the
    # reservoir temperature stays the fixed boundary value.
    np.testing.assert_allclose(
        network.node_temperature_of("outlet_manifold"),
        chan_1.coolant.node_fields.temperature[-1],
        rtol=1.0e-12,
    )
    assert network.node_temperature_of("recovery_bath") == 4.5


# ----------------------------------------------------------------------- #
# Stage 2: both channel ends coupled (pump loop)                           #
# ----------------------------------------------------------------------- #

INLET_PRESSURE = 600000.0  # CHAN_1/CHAN_2 inlet_pressure of CASE_1

# Ideal pump loop: a droop-free pump pins the supply manifold exactly
# head_at_zero_flow above the bath, and the near-ideal drain valve pins the
# return manifold at the bath pressure -- reproducing the INTIAL = 1
# imposed pressure drop.
IDEAL_PUMP_LOOP_SECTION = {
    "fluid_type": "helium",
    "nodes": [
        {
            "identifier": "bath",
            "kind": "reservoir",
            "pressure": OUTLET_PRESSURE,
            "temperature": 4.5,
        },
        {
            "identifier": "supply",
            "kind": "internal",
            "initial_pressure": INLET_PRESSURE,
            "initial_temperature": 4.5,
        },
        {
            "identifier": "return",
            "kind": "internal",
            "initial_pressure": OUTLET_PRESSURE,
            "initial_temperature": 4.5,
        },
    ],
    "branches": [
        {
            "identifier": "pump",
            "kind": "pump",
            "from": "bath",
            "to": "supply",
            "characteristic": {
                "head_at_zero_flow": INLET_PRESSURE - OUTLET_PRESSURE,
            },
        },
        {
            "identifier": "drain",
            "kind": "valve",
            "from": "return",
            "to": "bath",
            "linear_resistance": VALVE_RESISTANCE,
        },
    ],
    "ports": [
        {
            "node": "supply",
            "conductor": "CONDUCTOR_1",
            "channel": "CHAN_1",
            "end": "inlet",
        },
        {
            "node": "return",
            "conductor": "CONDUCTOR_1",
            "channel": "CHAN_1",
            "end": "outlet",
        },
    ],
}


def test_degenerate_pump_loop_matches_pressure_drop_boundary(tmp_path):
    """Both ends of CHAN_1 coupled: in the ideal-pump limit the loop must
    reproduce the INTIAL = 1 imposed-pressure-drop baseline."""
    baseline = run_simulation(
        prepare_run_directory(
            tmp_path,
            "baseline",
            conductor_edits=impose_pressure_drop_on_both_channels,
        )
    )
    coupled = run_simulation(
        prepare_run_directory(
            tmp_path,
            "coupled",
            network_section=IDEAL_PUMP_LOOP_SECTION,
            conductor_edits=impose_pressure_drop_on_both_channels,
        )
    )

    assert_conductors_match(baseline, coupled)

    # Loop consistency: the ideal pump pins the supply manifold and feeds
    # exactly the channel intake; the drain carries the discharge.
    network = coupled.hydraulic_network
    np.testing.assert_allclose(
        network.node_pressure_of("supply"), INLET_PRESSURE, rtol=1.0e-12
    )
    chan_1 = next(
        f_comp
        for f_comp in coupled.list_of_Conductors[0].inventory.fluids.collection
        if f_comp.identifier == "CHAN_1"
    )
    np.testing.assert_allclose(
        network.branch_mass_flow_of("pump"),
        chan_1.coolant.node_fields.mass_flow_rate[0],
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        network.branch_mass_flow_of("drain"),
        chan_1.coolant.node_fields.mass_flow_rate[-1],
        rtol=1.0e-3,
    )


# Pump loop with a bypass: the physical stage-2 configuration. BOTH
# channels are manifolded to the shared supply/return nodes -- the real
# CICC plumbing, and essential numerically: the two channels are openly
# connected along their whole length (hydraulic parallel), and a network
# that moved only one channel's plenum pressure would drive a strongly
# amplified differential between them (see
# warn_on_partially_ported_parallel_groups). The manifolds carry real
# compliance volumes so the loop relaxes smoothly from the declared
# initial pressures to its operating point, and the pump is sized to put
# that operating point near the committed 600 kPa supply state (total flow
# ~0.026 kg/s: two channel intakes ~0.021 plus ~0.005 through the bypass).
PUMP_HEAD = 2.015e4  # Pa
PUMP_DROOP = 1.5e7  # Pa/(kg/s)^2
BYPASS_RESISTANCE = 4.0e8  # Pa/(kg/s)^2
MANIFOLD_VOLUME = 2.0e-2  # m^3

BYPASS_LOOP_SECTION = {
    "fluid_type": "helium",
    "nodes": [
        {
            "identifier": "bath",
            "kind": "reservoir",
            "pressure": OUTLET_PRESSURE,
            "temperature": 4.5,
        },
        {
            "identifier": "supply",
            "kind": "internal",
            "initial_pressure": INLET_PRESSURE,
            "initial_temperature": 4.5,
            "volume": MANIFOLD_VOLUME,
        },
        {
            "identifier": "return",
            "kind": "internal",
            "initial_pressure": OUTLET_PRESSURE,
            "initial_temperature": 4.5,
            "volume": MANIFOLD_VOLUME,
        },
    ],
    "branches": [
        {
            "identifier": "pump",
            "kind": "pump",
            "from": "bath",
            "to": "supply",
            "characteristic": {
                "head_at_zero_flow": PUMP_HEAD,
                "quadratic_coefficient": PUMP_DROOP,
            },
        },
        {
            "identifier": "bypass",
            "kind": "valve",
            "from": "supply",
            "to": "return",
            "quadratic_resistance": BYPASS_RESISTANCE,
        },
        {
            "identifier": "drain",
            "kind": "valve",
            "from": "return",
            "to": "bath",
            "linear_resistance": 100.0,
        },
    ],
    "ports": [
        {
            "node": "supply",
            "conductor": "CONDUCTOR_1",
            "channel": channel,
            "end": "inlet",
        }
        for channel in ("CHAN_1", "CHAN_2")
    ] + [
        {
            "node": "return",
            "conductor": "CONDUCTOR_1",
            "channel": channel,
            "end": "outlet",
        }
        for channel in ("CHAN_1", "CHAN_2")
    ],
}

REDISTRIBUTION_END_TIME = 2.5


def boost_heat_pulse(conductor_document) -> None:
    """Retime the committed STR_MIX_1 heat pulse (500 W at 6-10 s over
    0.2 m) into the shortened window and strengthen it, so the channel
    impedance visibly rises during the run."""
    for component in conductor_document["components"]:
        if component["identifier"] == "STR_MIX_1":
            operations = component["operations"]
            operations["heat_flux_time_start"] = 0.3
            operations["heat_flux_time_end"] = 2.4
            operations["heat_flux_position_start"] = 0.5
            operations["heat_flux_position_end"] = 8.0
            operations["heat_flux_amplitude"] = 5000


# ----------------------------------------------------------------------- #
# Stage 3: several conductors coupled to one network                       #
# ----------------------------------------------------------------------- #

# Identifier renames of the cloned second conductor (component identifiers
# must be unique across the whole simulation).
SECOND_CONDUCTOR_RENAMES = {
    "CHAN_1": "CHAN_3",
    "CHAN_2": "CHAN_4",
    "STR_MIX_1": "STR_MIX_2",
    "Z_JACKET_1": "Z_JACKET_2",
}


def duplicate_conductor(run_directory: Path) -> None:
    """Clone the (already shortened) CONDUCTOR_1 input as CONDUCTOR_2 with
    renamed component identifiers, and register it in simulation.yaml. The
    two conductors have no thermal contact, so their physics stays
    independent unless a hydraulic network couples them."""
    conductor_document = yaml.safe_load(
        (run_directory / "conductor_CONDUCTOR_1.yaml").read_text()
    )
    conductor_document["conductor"]["identifier"] = "CONDUCTOR_2"
    for component in conductor_document["components"]:
        component["identifier"] = SECOND_CONDUCTOR_RENAMES[
            component["identifier"]
        ]
    conductor_document["coupling_component_order"] = [
        SECOND_CONDUCTOR_RENAMES.get(name, name)
        for name in conductor_document["coupling_component_order"]
    ]
    for coupling_record in conductor_document["couplings"]:
        coupling_record["between"] = [
            SECOND_CONDUCTOR_RENAMES.get(name, name)
            for name in coupling_record["between"]
        ]
    (run_directory / "conductor_CONDUCTOR_2.yaml").write_text(
        yaml.safe_dump(conductor_document, sort_keys=False)
    )
    simulation_document = yaml.safe_load(
        (run_directory / "simulation.yaml").read_text()
    )
    simulation_document["conductors"].append(
        {"file": "conductor_CONDUCTOR_2.yaml"}
    )
    (run_directory / "simulation.yaml").write_text(
        yaml.safe_dump(simulation_document, sort_keys=False)
    )


# Both conductors discharge one channel into the same manifold node, which
# drains through an almost ideal valve into a reservoir at the imposed
# outlet pressure: the degenerate limit of a genuinely multi-conductor
# network (the two banded systems meet in the shared Schur complement).
TWO_CONDUCTOR_NETWORK_SECTION = {
    "fluid_type": "helium",
    "nodes": [
        {
            "identifier": "shared_manifold",
            "kind": "internal",
            "initial_pressure": OUTLET_PRESSURE,
            "initial_temperature": 4.5,
        },
        {
            "identifier": "recovery_bath",
            "kind": "reservoir",
            "pressure": OUTLET_PRESSURE,
            "temperature": 4.5,
        },
    ],
    "branches": [
        {
            "identifier": "recovery_line",
            "kind": "valve",
            "from": "shared_manifold",
            "to": "recovery_bath",
            "linear_resistance": VALVE_RESISTANCE,
        },
    ],
    "ports": [
        {
            "node": "shared_manifold",
            "conductor": "CONDUCTOR_1",
            "channel": "CHAN_1",
            "end": "outlet",
        },
        {
            "node": "shared_manifold",
            "conductor": "CONDUCTOR_2",
            "channel": "CHAN_3",
            "end": "outlet",
        },
    ],
}


def test_two_conductors_share_a_network_node(tmp_path):
    """Two conductors coupled to one network through a shared manifold: in
    the degenerate limit both must match the fixed-boundary baseline, and
    the recovery line must carry the sum of both outlet flows."""
    baseline_directory = prepare_run_directory(tmp_path, "baseline")
    duplicate_conductor(baseline_directory)
    coupled_directory = prepare_run_directory(
        tmp_path, "coupled", network_section=TWO_CONDUCTOR_NETWORK_SECTION
    )
    duplicate_conductor(coupled_directory)

    baseline = run_simulation(baseline_directory)
    coupled = run_simulation(coupled_directory)

    for conductor_index in (0, 1):
        assert_conductors_match(baseline, coupled, conductor_index)

    # Shared-node mass balance: the recovery line carries the sum of the
    # two ported outlet flows (densities frozen one step back, hence the
    # modest tolerance against the freshly recomputed flows).
    network = coupled.hydraulic_network
    outlet_flow_sum = sum(
        next(
            f_comp
            for f_comp in conductor.inventory.fluids.collection
            if f_comp.identifier == channel
        ).coolant.node_fields.mass_flow_rate[-1]
        for conductor, channel in zip(
            coupled.list_of_Conductors, ("CHAN_1", "CHAN_3")
        )
    )
    np.testing.assert_allclose(
        network.branch_mass_flow_of("recovery_line"),
        outlet_flow_sum,
        rtol=1.0e-3,
    )

    # The network state file is written exactly once per step, into the
    # directory of the first coupled conductor.
    output_file = Path(
        coupled.dict_path["Output_Time_evolution_CONDUCTOR_1_dir"]
    ) / "hydraulic_network_te.tsv"
    frame = pd.read_csv(output_file, sep="\t")
    assert len(frame) == 21  # t = 0 plus 20 fixed steps of 0.1 s
    assert not (
        Path(coupled.dict_path["Output_Time_evolution_CONDUCTOR_2_dir"])
        / "hydraulic_network_te.tsv"
    ).exists()


def total_channel_intake(simulation: Simulation) -> float:
    """Combined mass flow drawn from the supply manifold by the coupled
    channels (positive into the channels), from the resolved ports."""
    return -sum(
        port.mass_flow_into_node()
        for port in simulation.list_of_Conductors[0].network_ports
        if port.end.value == "inlet"
    )


def test_heat_pulse_redistributes_and_pressurizes_the_loop(tmp_path):
    """The coupled physics no fixed-boundary run can reproduce: the strong
    heat pulse makes the channels expel hot fluid into the manifolds, the
    bypass takes over flow, and -- with the network-side energy transport
    of stage 4 -- the expelled enthalpy warms the manifolds, whose
    near-incompressible helium (thermal pressure coefficient beta/kappa_T
    of order 1e6 Pa/K) pressurizes the rigid loop by tens of kPa: the pump
    is pushed down (here: backward through) its droop curve and the drain
    vents the expansion surplus into the bath. Before stage 4 the static
    node temperatures silently discarded that enthalpy and the same pulse
    looked like a mild impedance-driven flow redistribution.

    The pulse effect is isolated by comparing against a control run with
    the identical pump loop but the committed (inert, t = 6-10 s) heat
    pulse: both runs share the mild settling transient of the loop."""
    def control_edits(conductor_document):
        impose_pressure_drop_on_both_channels(conductor_document)

    def pulsed_edits(conductor_document):
        impose_pressure_drop_on_both_channels(conductor_document)
        boost_heat_pulse(conductor_document)

    runs = {}
    for name, edits in (("control", control_edits), ("pulsed", pulsed_edits)):
        runs[name] = run_simulation(
            prepare_run_directory(
                tmp_path,
                name,
                network_section=BYPASS_LOOP_SECTION,
                end_time=REDISTRIBUTION_END_TIME,
                conductor_edits=edits,
            )
        )

    intake = {name: total_channel_intake(run) for name, run in runs.items()}
    bypass = {
        name: run.hydraulic_network.branch_mass_flow_of("bypass")
        for name, run in runs.items()
    }
    pump = {
        name: run.hydraulic_network.branch_mass_flow_of("pump")
        for name, run in runs.items()
    }

    assert intake["control"] > 0.0
    assert bypass["pulsed"] > bypass["control"]

    # Energy transport: the expelled hot fluid warms both manifolds -- the
    # return one through the ported outlets, the supply one through the
    # back-flowing inlets (observed at the end of the run: supply
    # 4.5 -> ~5.0 K, return -> ~5.2 K); the control supply only ever sees
    # the 4.5 K bath through the pump and must stay there.
    temperature = {
        name: {
            node: run.hydraulic_network.node_temperature_of(node)
            for node in ("supply", "return")
        }
        for name, run in runs.items()
    }
    np.testing.assert_allclose(
        temperature["control"]["supply"], 4.5, rtol=1.0e-9
    )
    assert temperature["pulsed"]["supply"] > 4.6
    assert temperature["pulsed"]["return"] > temperature["control"]["return"]

    # Thermal-expansion pressurization: the warming manifolds inflate the
    # loop pressure (observed: supply 600 -> ~639 kPa), the pump is pushed
    # down (backward through) its droop curve and the drain vents the
    # expansion surplus into the bath.
    supply_pressure = {
        name: run.hydraulic_network.node_pressure_of("supply")
        for name, run in runs.items()
    }
    assert supply_pressure["pulsed"] > supply_pressure["control"] + 1.0e4
    assert pump["pulsed"] < pump["control"]
    drain = {
        name: run.hydraulic_network.branch_mass_flow_of("drain")
        for name, run in runs.items()
    }
    assert drain["pulsed"] > drain["control"]

    # The network state time evolution is written next to the channel time
    # evolutions: header + one row per time level (t = 0 and 25 fixed steps
    # of 0.1 s), with the last row matching the in-memory final state.
    output_file = Path(
        runs["pulsed"].dict_path["Output_Time_evolution_CONDUCTOR_1_dir"]
    ) / "hydraulic_network_te.tsv"
    frame = pd.read_csv(output_file, sep="\t")
    network = runs["pulsed"].hydraulic_network
    assert list(frame.columns) == network.time_evolution_headers()
    assert len(frame) == 26
    np.testing.assert_allclose(
        frame["time (s)"].iloc[-1], REDISTRIBUTION_END_TIME, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        frame["pressure_supply (Pa)"].iloc[-1],
        supply_pressure["pulsed"],
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        frame["mass_flow_rate_bypass (kg/s)"].iloc[-1],
        bypass["pulsed"],
        rtol=1.0e-12,
    )
    assert (frame["pressure_bath (Pa)"] == OUTLET_PRESSURE).all()

    # The recorded return-manifold temperature matches the in-memory state
    # and the bath stays the fixed reference.
    np.testing.assert_allclose(
        frame["temperature_return (K)"].iloc[-1],
        temperature["pulsed"]["return"],
        rtol=1.0e-12,
    )
    assert (frame["temperature_bath (K)"] == 4.5).all()
