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


def prepare_run_directory(tmp_path, name: str, with_network: bool) -> Path:
    """Copy the CASE_1 YAML inputs and edit them into the shortened
    scenario, optionally adding the degenerate hydraulic network."""
    run_directory = tmp_path / name
    run_directory.mkdir()
    copy_input_files(CASE_1_INPUT_DIRECTORY, run_directory, "yaml")

    simulation_file = run_directory / "simulation.yaml"
    simulation_document = yaml.safe_load(simulation_file.read_text())
    simulation_document["simulation"]["end_time"] = SHORTENED_END_TIME
    if with_network:
        simulation_document["hydraulic_network"] = copy.deepcopy(
            NETWORK_SECTION
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
        if save_time <= SHORTENED_END_TIME
    ]
    conductor_file.write_text(
        yaml.safe_dump(conductor_document, sort_keys=False)
    )
    return run_directory


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


def test_degenerate_network_matches_fixed_pressure_boundary(tmp_path):
    baseline = run_simulation(
        prepare_run_directory(tmp_path, "baseline", with_network=False)
    )
    coupled = run_simulation(
        prepare_run_directory(tmp_path, "coupled", with_network=True)
    )

    baseline_conductor = baseline.list_of_Conductors[0]
    coupled_conductor = coupled.list_of_Conductors[0]

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
