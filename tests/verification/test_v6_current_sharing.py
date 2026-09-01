"""V6 — Self-consistent current sharing between a REBCO stack and copper.

One conductor with two electrically coupled strands: a REBCO STACK
driven above its critical current (I0 = 1.2 Ic) and a copper
STR_STAB, joined by a near-ideal contact conductance with
equipotential terminations. With ELECTRIC_CURRENT_CONSISTENCY the
steady electric solve iterates the strand resistances to the solved
currents; the converged split must match the uniform parallel-circuit
fixed point (power-law SC branch + two ohmic branches sharing one
longitudinal field E, analytical.current_sharing_reference).

A control run with the flag off pins the legacy semantics: the stack
resistance is built from the imposed current, so the solved split
differs from the self-consistent one.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source_code"))

import analytical  # noqa: E402

LENGTH = 10.0
MESH_ELEMENTS = 50
TEMPERATURE = 4.5
FIELD = 14.1

TAPE_COUNT = 10
STACK_WIDTH = 6.0e-3
TAPE_PITCH = 62.5e-6
HTS_THICKNESS = 2.0e-6
STACK_CROSS_SECTION = TAPE_COUNT * STACK_WIDTH * TAPE_PITCH
HTS_AREA = TAPE_COUNT * STACK_WIDTH * HTS_THICKNESS
STACK_STABILIZER_AREA = STACK_CROSS_SECTION - HTS_AREA
EXTERNAL_COPPER_AREA = 1.0e-4
RRR = 100
POWER_LAW_EXPONENT = 20
ELECTRIC_FIELD_CRITERION = 1.0e-4
C0 = 1.838522e8
TC0M = 93.0
BC20M = 140.0

CURRENT_OVER_IC = 1.2


def critical_current() -> float:
    from properties_of_materials.rare_earth_123 import (
        critical_current_density_re123,
    )

    jc = critical_current_density_re123(
        np.array([TEMPERATURE]), np.array([FIELD]), TC0M, BC20M, C0
    )[0]
    return jc * HTS_AREA


def copper_resistivity() -> float:
    from properties_of_materials.copper import electrical_resistivity_cu_nist

    return float(
        np.atleast_1d(
            electrical_resistivity_cu_nist(
                np.array([TEMPERATURE]), np.array([FIELD]), RRR
            )
        )[0]
    )


def write_run_directory(directory: Path, consistency: bool,
                        total_current: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "simulation.yaml").write_text(f"""format: opensc2-yaml/1
simulation:
  name: V6_CURRENT_SHARING
  end_time: 0.002
  time_stepping:
    adaptivity: ERROR_CONTROLLED
    minimum_step: 0.001
    maximum_step: 0.001
    tolerance: 0.001
    reference_time: 0
    reference_duration: 10
environment:
  medium: Air
  temperature: 300.15
  pressure: 101325
conductors:
- file: conductor_CONDUCTOR_100.yaml
""")
    (directory / "external_current.csv").write_text(
        f"STACK1_V1,0.0,0,1e+06\n"
        f"STACK1_V1,0.0,{total_current:g},{total_current:g}\n"
    )

    strand_operations_common = f"""    operating_current_interpolation: linear
    fixed_potential_flag: 1
    fixed_potential_number: 1
    fixed_potential_coordinates: 0
    fixed_potential_values: 0
    magnetic_field_mode: 0
    magnetic_field_interpolation: linear
    magnetic_field_units: T
    magnetic_field_inlet_initial: {FIELD}
    magnetic_field_outlet_initial: {FIELD}
    magnetic_field_inlet_transient: {FIELD}
    magnetic_field_outlet_transient: {FIELD}
    field_angle_mode: 0
    field_angle_interpolation: linear
    strain_mode: 0
    strain_interpolation: linear
    strain_value: 0
    heat_flux_mode: 0
    heat_flux_interpolation: linear
    heat_flux_position_start: 0
    heat_flux_position_end: 0
    heat_flux_amplitude: 0
    heat_flux_time_start: 0
    heat_flux_time_end: 0
    initial_temperature_mode: 0
    inlet_temperature: {TEMPERATURE}
    outlet_temperature: {TEMPERATURE}
    current_sharing_temperature_evaluation: 0"""

    (directory / "conductor_CONDUCTOR_100.yaml").write_text(f"""conductor:
  name: CONDUCTOR
  identifier: CONDUCTOR_100
  inputs:
    length: {LENGTH}
    diameter: 0.03
    is_rectangular: 0
    width: 0
    height: 0
    current_mode: -1
    initial_current: {total_current}
    is_joint: 0
    inlet_heated_zone_start: 0
    inlet_heated_zone_end: 0
    outlet_heated_zone_start: 0
    outlet_heated_zone_end: 0
    thermohydraulic_method: BDF2
    upwind: 1
    external_free_convection_correlation: vertical_plate_churchill_chu
    phi_radiative: 0
    phi_convective: 0
    electric_method: BE
    electric_time_step: 0.1
  operations:
    do_equipotential_surfaces_exist: 1
    number_of_equipotential_surfaces: 2
    equipotential_surface_coordinates: [0.0, {LENGTH}]
    maximum_iteration_number: 100
    inductance_mode: 1
    self_inductance_mode: 2
    electric_solver: 0
    electric_current_consistency: {str(consistency).lower()}
components:
- identifier: CHAN1_V1
  kind: CHAN
  sheet: CHAN
  inputs:
    cross_section: 1.9634954e-05
    x_barycenter: 0
    y_barycenter: 0
    fluid_type: helium
    hydraulic_diameter: 0.005
    roughness: 0
    cos_theta: 1
    void_fraction: 1.0
    friction_factor_model: 110
    friction_multiplier: 1.0
    is_rectangular: 0
    width: 0
    height: 0
    channel_type: bundle
    heat_transfer_model: 2
    show_figure: 0
  operations:
    hydraulic_boundary_condition: 1
    boundary_values_from_file: false
    inlet_temperature: {TEMPERATURE}
    outlet_temperature: {TEMPERATURE}
    initial_temperature: {TEMPERATURE}
    initial_temperature_outlet: {TEMPERATURE}
    inlet_pressure: 600000.0
    outlet_pressure: 599000.0
    initial_pressure: 600000.0
    inlet_mass_flow_rate: 0.0005
    outlet_mass_flow_rate: 0.0005
    flow_direction: forward
- identifier: STACK1_V1
  kind: STACK
  sheet: STACK
  inputs:
    cross_section: {STACK_CROSS_SECTION}
    x_barycenter: 0.008
    y_barycenter: 0
    cos_theta: 1.0
    superconducting_material: YBCO
    critical_current_scaling_constant: {C0}
    upper_critical_field_at_zero_temperature: {BC20M}
    critical_temperature_at_zero_field: {TC0M}
    power_law_exponent: {POWER_LAW_EXPONENT}
    electric_field_criterion: {ELECTRIC_FIELD_CRITERION}
    residual_resistivity_ratio: {RRR}
    stack_width: {STACK_WIDTH}
    tape_identifier: {TAPE_COUNT}
    tape_count: {TAPE_COUNT}
    number_of_material_layers: 4
    HTS_material: re123
    HTS_thickness: {HTS_THICKNESS}
    silver_material: ag
    silver_thickness: 2.0e-06
    substrate_material: hc276
    substrate_thickness: 4.0e-05
    stabilizer_material: cu
    stabilizer_thickness: 1.85e-05
    show_figure: 0
  operations:
    operating_current_mode: -1
{strand_operations_common}
- identifier: STR_STAB1_V1
  kind: STR_STAB
  sheet: STR_STAB
  inputs:
    cross_section: {EXTERNAL_COPPER_AREA}
    x_barycenter: -0.008
    y_barycenter: 0
    cos_theta: 1.0
    stabilizer_material: Cu
    residual_resistivity_ratio: {RRR}
    show_figure: 0
  operations:
    operating_current_mode: none
{strand_operations_common}
coupling_component_order:
- Environment
- CHAN1_V1
- STACK1_V1
- STR_STAB1_V1
couplings:
- between:
  - CHAN1_V1
  - STR_STAB1_V1
  contact_perimeter_mode: 1
  contact_perimeter: 0.015708
  heat_transfer_coefficient_mode: 2
  heat_transfer_coefficient_multiplier: 1
- between:
  - STACK1_V1
  - STR_STAB1_V1
  contact_perimeter_mode: 1
  contact_perimeter: 0.006
  heat_transfer_coefficient_mode: 1
  thermal_contact_resistance: 1.0e-05
  heat_transfer_coefficient_multiplier: 1
  electric_conductance_mode: 1
  electric_conductance: 1.0e8
grid:
  number_of_elements: {MESH_ELEMENTS}
  mesh_type: 0
  refined_zone_number_of_elements: 0
  refined_zone_start: 0
  refined_zone_end: 0
  minimum_element_size: 0.0001
  maximum_element_size: 1.0
  growth_ratio_left: 1.2
  growth_ratio_right: 1.2
  maximum_number_of_nodes: {MESH_ELEMENTS + 20}
diagnostics:
  spatial_distribution_times:
  - 0.001
  time_evolution_positions:
  - 0.0
  - {LENGTH / 2.0}
  - {LENGTH}
""")


class CurrentCapture:
    """Step callback: record mid-length solved currents per strand."""

    def __init__(self):
        self.stack_current = None
        self.external_current = None

    def __call__(self, simulation) -> bool:
        conductor = simulation.list_of_Conductors[0]
        for strand in conductor.inventory.strands.collection:
            mid = strand.gauss_fields.current_along.size // 2
            value = float(np.real(strand.gauss_fields.current_along[mid]))
            if strand.identifier.startswith("STACK"):
                self.stack_current = value
            else:
                self.external_current = value
        return False


def run_case(directory: Path, consistency: bool, total_current: float):
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    from simulation import Simulation

    write_run_directory(directory, consistency, total_current)
    capture = CurrentCapture()
    Simulation(str(directory) + "/", step_callback=capture).run()
    return capture


@pytest.fixture(scope="module")
def reference():
    total_current = CURRENT_OVER_IC * critical_current()
    return total_current, analytical.current_sharing_reference(
        total_current=total_current,
        critical_current=critical_current(),
        power_law_exponent=POWER_LAW_EXPONENT,
        electric_field_criterion=ELECTRIC_FIELD_CRITERION,
        stack_stabilizer_area=STACK_STABILIZER_AREA,
        external_stabilizer_area=EXTERNAL_COPPER_AREA,
        copper_resistivity=copper_resistivity(),
    )


def test_consistent_split_matches_parallel_circuit(tmp_path, reference):
    total_current, expected = reference
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capture = run_case(tmp_path / "consistent", True, total_current)
    cap_warnings = [
        w for w in caught
        if "current-consistency" in str(w.message)
    ]
    assert not cap_warnings, f"iteration hit the cap: {cap_warnings[0].message}"

    assert capture.stack_current is not None
    # 2 % tolerance: one thermal step of Joule heating shifts the copper
    # resistivity slightly off the 4.5 K reference value.
    assert capture.stack_current == pytest.approx(
        expected["stack_current"], rel=0.02
    )
    assert capture.external_current == pytest.approx(
        expected["external_current"], rel=0.05
    )
    # Kirchhoff: the two strands together carry the injected current.
    assert capture.stack_current + capture.external_current == pytest.approx(
        total_current, rel=1e-6
    )


def test_legacy_flag_off_is_not_self_consistent(tmp_path, reference):
    total_current, expected = reference
    capture = run_case(tmp_path / "legacy", False, total_current)

    assert capture.stack_current is not None
    # Legacy semantics: stack resistance built from the imposed current
    # (I0), not the solved one - the split misses the consistent fixed
    # point by more than the consistent-run tolerance.
    assert capture.stack_current != pytest.approx(
        expected["stack_current"], rel=0.005
    )
