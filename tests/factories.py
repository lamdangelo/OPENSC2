"""Factories that build real OPENSC2 objects for unit tests.

Components are created through their public constructors. The inputs and
operations dataclasses default to values taken from the example decks
(CASE_1: Nb3Sn strand and steel/G10 jacket; CASE_2: aluminium stabilizer
strand; HELIAS-VIPER: REBCO stack) and every field can be overridden by
keyword. :func:`make_conductor` returns a light conductor stub that owns the
same state objects as ``conductor.Conductor`` (``time_state``, ``electric``,
``balances``, ``save_schedule``), a straight mesh and a real component
inventory, so tests never spell conductor state the legacy way.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from components.component_build import ComponentBuildSettings
from components.component_collection import ComponentInventory
from components.component_flags import ComponentType
from components.jacket.jacket_component import JacketComponent
from components.jacket.jacket_component_inputs import JacketComponentInputs
from components.solid.solid_component_inputs import (
    SolidComponentOperations,
    StrandComponentOperations,
)
from components.solid.stack_component import StackComponent
from components.solid.stack_component_inputs import StackComponentInputs, TapeLayer
from components.solid.strand_mixed_component import StrandMixedComponent
from components.solid.strand_mixed_component_inputs import StrandMixedComponentInputs
from components.solid.strand_stabilizer_component import StrandStabilizerComponent
from components.solid.strand_stabilizer_component_inputs import (
    StrandStabilizerComponentInputs,
)
from conductor.conductor import Conductor
from conductor.conductor_flags import InterpolationType, MethodFlag
from conductor.electric_state import ElectricState
from conductor.run_state import UNSET, BalanceState, SaveSchedule, TimeState
from electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode
from hydraulics.hydraulic_flags import HydraulicFormulation
from runtime.time_stepping import TimeSteppingSettings
from thermal.thermal_flags import HeatExcitation


# --------------------------------------------------------------------------
# Build settings
# --------------------------------------------------------------------------

class NullExternalInputReader:
    """ports.external_inputs.ExternalInputReader for components without
    auxiliary files: every read fails loudly."""

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError("this test component has no external input files")

    tabulated_series = _unavailable
    scaling_input = _unavailable
    read_flow_parameters = _unavailable
    user_defined_quantity = _unavailable
    user_defined_grid_tables = _unavailable


def build_settings(
    conductor_identifier="CONDUCTOR_1",
    conductor_length=10.0,
    adaptive_mode=0,
    minimum_time_step=1.0e-3,
    external_inputs=None,
) -> ComponentBuildSettings:
    return ComponentBuildSettings(
        conductor_identifier=conductor_identifier,
        conductor_length=conductor_length,
        time_stepping=TimeSteppingSettings(
            adaptive_mode=adaptive_mode, minimum_time_step=minimum_time_step
        ),
        external_inputs=external_inputs or NullExternalInputReader(),
    )


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

_SOLID_OPERATIONS = dict(
    magnetic_field_bc_mode=BFieldDefinitionType.CONSTANT_OR_LINEAR,
    magnetic_field_units=None,
    magnetic_field_inlet_initial=12.0,
    magnetic_field_outlet_initial=12.0,
    magnetic_field_inlet_transient=0.0,
    magnetic_field_outlet_transient=0.0,
    magnetic_field_interpolation=InterpolationType.LINEAR,
    operating_current_mode=CurrentMode.CURRENT_IS_CONSTANT,
    operating_current_interpolation=InterpolationType.LINEAR,
    heat_flux_mode=HeatExcitation.NO_HEATING,
    heat_flux_time_start=0.0,
    heat_flux_time_end=0.0,
    heat_flux_amplitude=0.0,
    heat_flux_position_start=0.0,
    heat_flux_position_end=0.0,
    heat_flux_interpolation=InterpolationType.LINEAR,
    initial_temperature_mode=0,
    inlet_temperature=0.0,
    outlet_temperature=0.0,
)

_STRAND_OPERATIONS = dict(
    alpha_b_mode=0,
    alpha_b_interpolation=InterpolationType.LINEAR,
    strain_mode=0,
    strain_value=0.0,
    tcs_evaluation=False,
    fix_potential_flag=False,
    fix_potential_number=0,
    fix_potential_coordinate=0,
    fix_potential_value=0,
)


def solid_operations(**overrides) -> SolidComponentOperations:
    """Jacket-style operations (no heating, constant field)."""
    return SolidComponentOperations(**{**_SOLID_OPERATIONS, **overrides})


def strand_operations(**overrides) -> StrandComponentOperations:
    """Strand/stack operations (no heating, constant field, no fixed potential)."""
    return StrandComponentOperations(
        **{**_SOLID_OPERATIONS, **_STRAND_OPERATIONS, **overrides}
    )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

_STRAND_GEOMETRY_KEYS = {
    "num_sc_strands",
    "sc_strand_diameter",
    "num_stabilizer_strands",
    "stabilizer_strand_diameter",
}


def strand_mixed_inputs(**overrides) -> StrandMixedComponentInputs:
    """CASE_1 ITER-like Nb3Sn/Cu strand bundle.

    Overriding ``cross_section`` without the strand geometry turns the
    bundle into one superconducting strand filling that cross section
    (``stabilizer_to_sc_ratio`` then splits it, mode 0), so the constructor's
    consistency check passes.
    """
    values = dict(
        cross_section=0.000753951765,
        cos_theta=0.9699,
        x_barycenter=0.0,
        y_barycenter=0.0,
        show_figure=False,
        superconducting_material="nb3sn",
        upper_critical_field_at_0K=32.35,
        critical_current_scaling_constant=1.21508e11,
        critical_temperature_at_0T=16.22,
        stabilizer_material="cu",
        num_sc_strands=900,
        sc_strand_diameter=0.000822,
        num_stabilizer_strands=522,
        stabilizer_strand_diameter=0.000821,
        stabilizer_to_sc_ratio_mode=1,
        stabilizer_to_sc_ratio=2.0846,
        critical_current_definition_mode=0,
        num_material_types=2,
        flux_flow_electric_field=1.0e-5,
        power_law_exponent=5.0,
        residual_resistivity_ratio=197.71,
    )
    if "cross_section" in overrides and not (_STRAND_GEOMETRY_KEYS & overrides.keys()):
        values.update(
            num_sc_strands=1,
            sc_strand_diameter=float(np.sqrt(4.0 * overrides["cross_section"] / np.pi)),
            num_stabilizer_strands=0,
            stabilizer_strand_diameter=0.0,
            stabilizer_to_sc_ratio_mode=0,
        )
    values.update(overrides)
    return StrandMixedComponentInputs(**values)


def stack_inputs(**overrides) -> StackComponentInputs:
    """HELIAS-VIPER REBCO stack: 80 tapes, 6 mm wide, 2 um of REBCO each."""
    values = dict(
        cross_section=3.0e-5,
        cos_theta=1.0,
        x_barycenter=0.0,
        y_barycenter=0.0,
        show_figure=False,
        superconducting_material="ybco",
        upper_critical_field_at_0K=140.0,
        critical_current_scaling_constant=1.838522e8,
        critical_temperature_at_0T=93.0,
        stack_width=6.0e-3,
        tape_identifier="80",
        num_material_layers=4,
        num_tapes=80,
        residual_resistivity_ratio=100.0,
        flux_flow_electric_field=1.0e-4,
        power_law_exponent=20.0,
        tape_layers=[
            TapeLayer("HTS", "re123", 2.0e-6, is_hts=True),
            TapeLayer("silver", "ag", 2.0e-6),
            TapeLayer("substrate", "hc276", 4.0e-5),
            TapeLayer("stabilizer", "cu", 1.85e-5),
        ],
    )
    values.update(overrides)
    return StackComponentInputs(**values)


def strand_stabilizer_inputs(**overrides) -> StrandStabilizerComponentInputs:
    """CASE_2 aluminium stabilizer strand."""
    values = dict(
        cross_section=8.608e-06,
        cos_theta=0.9699,
        x_barycenter=0.0,
        y_barycenter=0.0,
        show_figure=False,
        stabilizer_material="al",
        residual_resistivity_ratio=None,
    )
    values.update(overrides)
    return StrandStabilizerComponentInputs(**values)


def jacket_inputs(**overrides) -> JacketComponentInputs:
    """CASE_1 stainless-steel jacket with G10 insulation."""
    values = dict(
        cross_section=0.00058665,
        cos_theta=1.0,
        x_barycenter=0.0,
        y_barycenter=0.0,
        show_figure=False,
        jacket_material="ss",
        insulation_material="ge",
        jacket_cross_section=0.00030699,
        insulation_cross_section=0.00027966,
        num_material_types=2,
        jacket_kind="whole_enclosure",
        emissivity=1.0,
        outer_perimeter=0.1357,
        inner_perimeter=0.1288,
    )
    values.update(overrides)
    return JacketComponentInputs(**values)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

def make_strand_mixed(identifier="STR_MIX_1", operations=None, build=None, **input_overrides):
    return StrandMixedComponent(
        identifier,
        ComponentType.STRAND_MIXED.value,
        strand_mixed_inputs(**input_overrides),
        operations or strand_operations(),
        build or build_settings(),
    )


def make_stack(identifier="STACK_1", operations=None, build=None, **input_overrides):
    return StackComponent(
        identifier,
        ComponentType.STACK.value,
        stack_inputs(**input_overrides),
        operations or strand_operations(),
        build or build_settings(),
    )


def make_strand_stabilizer(identifier="STR_STAB_1", operations=None, build=None, **input_overrides):
    return StrandStabilizerComponent(
        identifier,
        ComponentType.STRAND_STABILIZER.value,
        strand_stabilizer_inputs(**input_overrides),
        operations or strand_operations(),
        build or build_settings(),
    )


def make_jacket(identifier="Z_JACKET_1", operations=None, build=None, **input_overrides):
    return JacketComponent(
        identifier,
        ComponentType.JACKET.value,
        jacket_inputs(**input_overrides),
        operations or solid_operations(),
        build or build_settings(),
    )


# --------------------------------------------------------------------------
# Conductor
# --------------------------------------------------------------------------

class ConductorStub:
    """What the physics reads from a Conductor, without input files.

    Owns the same state objects as ``conductor.Conductor``. Transitional:
    the legacy attribute spelling (``cond_time``, ``time_step``,
    ``nodal_potential``, ...) is forwarded to the state objects exactly as
    ``Conductor._FORWARDED`` does, until the production code reads the
    state objects directly.
    """

    def __init__(self, identifier, mesh, inventory, inputs, operations, file_paths, time, time_step):
        put = object.__setattr__
        put(self, "electric", ElectricState())
        put(self, "time_state", TimeState())
        put(self, "balances", BalanceState())
        put(self, "save_schedule", SaveSchedule())
        put(self, "identifier", identifier)
        put(self, "mesh", mesh)
        put(self, "inventory", inventory)
        put(self, "inputs", inputs)
        put(self, "operations", operations)
        put(self, "file_paths", file_paths)
        put(self, "network_ports", [])
        put(self, "hydraulic_formulation", HydraulicFormulation.VELOCITY)
        put(self, "fluid_source_callback", None)
        self.time_state.cond_time = [time]
        self.time_state.cond_num_step = 0
        self.time_state.time_step = time_step
        self.time_state.previous_time_step = time_step

    def component_element_lengths(self, identifier: str) -> np.ndarray:
        """Straight components: every element has the mesh element length."""
        return np.asarray(self.mesh.element_lengths, dtype=float)

    # -- transitional forwarding of the legacy spelling (see Conductor) ----
    def __getattr__(self, name):
        holder_name = Conductor._FORWARDED.get(name)
        if holder_name is not None and holder_name in self.__dict__:
            value = getattr(self.__dict__[holder_name], name)
            if value is not UNSET:
                return value
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        holder_name = Conductor._FORWARDED.get(name)
        if holder_name is not None and holder_name in self.__dict__:
            setattr(self.__dict__[holder_name], name, value)
        else:
            object.__setattr__(self, name, value)


def make_conductor(
    number_of_nodes=11,
    length=10.0,
    time=0.0,
    time_step=1.0e-3,
    identifier="CONDUCTOR_1",
    components=(),
    **attributes,
) -> ConductorStub:
    """A conductor stub with a straight uniform mesh.

    ``components`` are added to a real ``ComponentInventory``; any further
    keyword overrides or adds an attribute (``mesh``, ``inventory``,
    ``inputs``, ``operations``, ``file_paths``, ``network_ports``, ...).
    """
    coordinates = np.linspace(0.0, length, number_of_nodes)
    mesh = SimpleNamespace(
        number_of_nodes=number_of_nodes,
        number_of_elements=number_of_nodes - 1,
        node_coordinates=coordinates,
        element_lengths=np.diff(coordinates),
        gauss_point_coordinates=0.5 * (coordinates[:-1] + coordinates[1:]),
        position_precision=10,
    )
    inventory = ComponentInventory.empty()
    for component in components:
        inventory.add(component)
    stub = ConductorStub(
        identifier=identifier,
        mesh=mesh,
        inventory=inventory,
        inputs=SimpleNamespace(
            zlength=length,
            thermohydraulic_method=MethodFlag.BACKWARD_EULER,
            current_mode=CurrentMode.CURRENT_NOT_DEFINED,
            initial_current=0.0,
        ),
        operations=SimpleNamespace(
            electric_resistance_floor=0.0, electric_current_consistency=False
        ),
        file_paths=SimpleNamespace(base_path=None, structure_elements_path=None),
        time=time,
        time_step=time_step,
    )
    for name, value in attributes.items():
        setattr(stub, name, value)
    return stub
