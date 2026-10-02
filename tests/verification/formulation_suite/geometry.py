"""Shared geometry of the formulation verification suite.

Conductor definition: the W7-X NbTi CICC of
``W7-X/comparison_1D/conductor_CONDUCTOR_100.yaml`` (one bundle channel,
friction model 206 Darcy-Forchheimer porous medium, mixed NbTi/Cu strand,
Al6063 jacket, glass-epoxy insulation), patched per conductor: length, mesh,
no electric problem (``current_mode: none``), constant magnetic field, losses
off, pressure-drop boundary condition at 5.0 / 4.5 bar and 4.5 K.

Network: supply reservoir -> linear valve (R_far) -> supply volume (V) ->
inlets of N conductors; outlets -> return volume -> linear valve -> return
reservoir.

The module also exposes the code's own friction law and the helium property
library as plain callables for the reference solvers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

import interfaces.coolprop_interface as cpi
from components.fluid.fluid_component_inputs import FluidComponentInputs
from hydraulics.friction_factor_factory import FrictionFactorFactory
from hydraulics.friction_factor_models import FrictionFactorModelType
from hydraulics.hydraulic_flags import get_channel_type, get_fluid_type, FluidType
from thermal.heat_transfer_models import HeatTransferModelType

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DECK = REPOSITORY_ROOT / "W7-X" / "comparison_1D" / "conductor_CONDUCTOR_100.yaml"

# Operating point.
SUPPLY_PRESSURE = 5.0e5  # Pa
RETURN_PRESSURE = 4.5e5  # Pa
TEMPERATURE = 4.5  # K
LENGTHS = (20.0, 30.0, 40.0)  # m, L1 : L2 : L3 = 1 : 1.5 : 2
VOLUME = 5.0e-4  # m^3 (0.5 L) per volume node
# R_far = FAR_RESISTANCE_FACTOR * Z_line. The plan's default of 10 Z_line was
# rejected by the numbers: with Z_line = c/A ~ 6e6 Pa/(kg/s) the two valves
# would drop 2 x 0.25 bar of the 0.5 bar and leave the channels at Mach
# 1.5e-4 instead of O(1e-3). With R_far = Z_line the DC drop (0.3 bar per
# valve at Q ~ 5 g/s) is compensated by offsetting the reservoir pressures
# (supply + R Q, return - R Q, see tiers.operating_point) so that the volume
# nodes sit at 5.0 / 4.5 bar, and the AC termination Z_t = R || 1/(i omega C)
# runs from a matched load (Gamma = 0) at DC to Gamma -> -1 at high
# frequency: the capacitive behaviour of the volume is genuinely tested.
FAR_RESISTANCE_FACTOR = 1.0

PROPERTY_ALIASES = dict(
    density="Dmass",
    isothermal_compressibility="isothermal_compressibility",
    isobaric_expansion_coefficient="isobaric_expansion_coefficient",
    isochoric_specific_heat="Cvmass",
    isobaric_specific_heat="Cpmass",
    speed_of_sound="speed_of_sound",
    viscosity="viscosity",
    enthalpy="Hmass",
)


def helium_properties(temperature, pressure) -> dict:
    """Property-library wrapper (production path, tabulated) as a callable
    for the reference solvers; scalars in -> floats out, arrays -> arrays."""
    scalar = np.isscalar(temperature) and np.isscalar(pressure)
    result = cpi.compute_properties(
        FluidType.HELIUM, PROPERTY_ALIASES, np.atleast_1d(temperature), np.atleast_1d(pressure)
    )
    if scalar:
        return {key: float(np.ravel(value)[0]) for key, value in result.items()}
    return {key: np.asarray(value, dtype=float) for key, value in result.items()}


def load_template() -> dict:
    return yaml.safe_load(TEMPLATE_DECK.read_text())


def template_channel(document: dict) -> dict:
    return next(c for c in document["components"] if c["kind"] == "CHAN")


def channel_inputs(document: dict) -> FluidComponentInputs:
    """FluidComponentInputs of the template channel (for the code's friction law)."""
    inputs = template_channel(document)["inputs"]
    return FluidComponentInputs(
        cross_section=float(inputs["cross_section"]),
        x_barycenter=float(inputs["x_barycenter"]),
        y_barycenter=float(inputs["y_barycenter"]),
        cos_theta=float(inputs["cos_theta"]),
        void_fraction=float(inputs["void_fraction"]),
        hydraulic_diameter=float(inputs["hydraulic_diameter"]),
        roughness=float(inputs["roughness"]),
        is_rectangular=bool(inputs["is_rectangular"]),
        width=float(inputs["width"]),
        height=float(inputs["height"]),
        fluid_type=get_fluid_type(str(inputs["fluid_type"])),
        friction_factor_model=FrictionFactorModelType.get_friction_factor_model(
            int(inputs["friction_factor_model"])
        ),
        heat_transfer_model=HeatTransferModelType.get_heat_transfer_model(
            int(inputs["heat_transfer_model"])
        ),
        friction_multiplier=float(inputs["friction_multiplier"]),
        channel_type=get_channel_type(str(inputs["channel_type"])),
        show_figure=False,
    )


def code_friction_law(inputs: FluidComponentInputs):
    """Total Fanning friction factor f(Re) exactly as Channel.evaluate_friction_factors."""
    models = FrictionFactorFactory.create(inputs)

    def friction_factor(reynolds):
        reynolds = np.abs(np.asarray(reynolds, dtype=float))
        laminar = models.laminar(reynolds)
        turbulent = models.turbulent(reynolds)
        total = models.total(reynolds, laminar, turbulent) * inputs.friction_multiplier
        return float(total) if np.ndim(total) == 0 else total

    return friction_factor


@dataclass
class ConductorSpec:
    index: int  # 1-based
    length: float
    elements: int

    @property
    def identifier(self) -> str:
        return f"CONDUCTOR_{self.index}"

    @property
    def channel(self) -> str:
        return f"CHAN1_C{self.index}"

    @property
    def strand(self) -> str:
        return f"STR_MIX1_C{self.index}"


@dataclass
class NetworkCase:
    conductors: list
    time_step: float
    end_time: float
    formulation: str  # "velocity" | "mass_flow"
    supply_pressure: float = SUPPLY_PRESSURE
    return_pressure: float = RETURN_PRESSURE
    temperature: float = TEMPERATURE
    volume: float = VOLUME
    far_resistance: float = 0.0
    pressure_supply_volume: float = SUPPLY_PRESSURE
    pressure_return_volume: float = RETURN_PRESSURE
    mass_flows: dict = field(default_factory=dict)  # identifier -> initial mdot
    heated_strands: dict = field(default_factory=dict)  # conductor identifier -> True
    spatial_times: list = field(default_factory=list)
    method: str = "BDF2"
    name: str = "formulation_suite"
    heat_transfer_model: int | None = None  # None: the template's (2, Dittus-Boelter pure)


def _rename(document: dict, index: int) -> dict:
    """Rename every component of the template with the conductor suffix,
    propagating into the coupling order and the couplings."""
    mapping = {}
    for component in document["components"]:
        old = component["identifier"]
        new = old.replace("_D1", f"_C{index}")
        mapping[old] = new
        component["identifier"] = new
    document["coupling_component_order"] = [
        mapping.get(name, name) for name in document["coupling_component_order"]
    ]
    for coupling in document["couplings"]:
        coupling["between"] = [mapping.get(name, name) for name in coupling["between"]]
    return mapping


def build_conductor_document(template: dict, spec: ConductorSpec, case: NetworkCase,
                             mass_flow: float, heated: bool) -> dict:
    document = copy.deepcopy(template)
    _rename(document, spec.index)
    conductor = document["conductor"]
    conductor["identifier"] = spec.identifier
    inputs = conductor["inputs"]
    inputs["length"] = float(spec.length)
    inputs["current_mode"] = "none"
    inputs["initial_current"] = 0.0
    inputs["thermohydraulic_method"] = case.method
    if case.formulation == "velocity":
        inputs["hydraulic_formulation"] = "velocity"
        inputs["explicit_mass_flow_formulation"] = False
    elif case.formulation == "mass_flow":
        inputs["hydraulic_formulation"] = "mass_flow"
        inputs["explicit_mass_flow_formulation"] = True
    else:
        raise ValueError(case.formulation)
    for component in document["components"]:
        operations = component["operations"]
        if component["kind"] == "CHAN":
            if case.heat_transfer_model is not None:
                component["inputs"]["heat_transfer_model"] = int(case.heat_transfer_model)
            operations["hydraulic_boundary_condition"] = 1
            operations["boundary_values_from_file"] = False
            operations["inlet_temperature"] = case.temperature
            operations["outlet_temperature"] = case.temperature
            operations["initial_temperature"] = case.temperature
            operations["initial_temperature_outlet"] = case.temperature
            operations["inlet_pressure"] = float(case.pressure_supply_volume)
            operations["outlet_pressure"] = float(case.pressure_return_volume)
            operations["initial_pressure"] = float(case.pressure_supply_volume)
            operations["inlet_mass_flow_rate"] = float(mass_flow)
            operations["outlet_mass_flow_rate"] = float(mass_flow)
            operations["flow_direction"] = "forward"
        else:
            operations["operating_current_mode"] = "none"
            operations["magnetic_field_mode"] = 1
            operations["magnetic_field_inlet_initial"] = 0.0
            operations["magnetic_field_outlet_initial"] = 0.0
            operations["inlet_temperature"] = case.temperature
            operations["outlet_temperature"] = case.temperature
            operations["initial_temperature_mode"] = 0
            operations["heat_flux_mode"] = 0
            operations["heat_flux_amplitude"] = 0.0
            for key in ("coupling_loss_time_constant", "eddy_loss_geometry_constant",
                        "filament_diameter"):
                if key in operations:
                    operations[key] = 0.0
            if "magnetic_field_scales_with_current" in operations:
                operations["magnetic_field_scales_with_current"] = False
            if "transverse_coupling_file" in operations:
                operations["transverse_coupling_file"] = ""
            if component["kind"] == "STR_MIX" and heated:
                operations["heat_flux_mode"] = -2  # user function (heat_pulse.py)
    document["grid"] = {
        "number_of_elements": int(spec.elements),
        "mesh_type": 0,
        "refined_zone_number_of_elements": 0,
        "refined_zone_start": 0,
        "refined_zone_end": 0,
        "minimum_element_size": 1.0e-4,
        "maximum_element_size": 10.0,
        "growth_ratio_left": 1.2,
        "growth_ratio_right": 1.2,
        "maximum_number_of_nodes": int(spec.elements) + 50,
    }
    document["diagnostics"] = {
        "spatial_distribution_times": [float(t) for t in case.spatial_times],
        "time_evolution_positions": [0.0, round(spec.length / 2.0, 6), float(spec.length)],
    }
    return document


def network_section(case: NetworkCase) -> dict:
    total = float(sum(case.mass_flows.values()))
    ports = []
    for spec in case.conductors:
        ports.append({"node": "supply_volume", "conductor": spec.identifier,
                      "channel": spec.channel, "end": "inlet"})
        ports.append({"node": "return_volume", "conductor": spec.identifier,
                      "channel": spec.channel, "end": "outlet"})
    return {
        "fluid_type": "helium",
        "nodes": [
            {"identifier": "supply_bc", "kind": "reservoir",
             "pressure": float(case.supply_pressure), "temperature": float(case.temperature)},
            {"identifier": "supply_volume", "kind": "internal",
             "initial_pressure": float(case.pressure_supply_volume),
             "initial_temperature": float(case.temperature), "volume": float(case.volume)},
            {"identifier": "return_volume", "kind": "internal",
             "initial_pressure": float(case.pressure_return_volume),
             "initial_temperature": float(case.temperature), "volume": float(case.volume)},
            {"identifier": "return_bc", "kind": "reservoir",
             "pressure": float(case.return_pressure), "temperature": float(case.temperature)},
        ],
        "branches": [
            {"identifier": "supply_valve", "kind": "valve", "from": "supply_bc",
             "to": "supply_volume", "linear_resistance": float(case.far_resistance),
             "initial_mass_flow": total},
            {"identifier": "return_valve", "kind": "valve", "from": "return_volume",
             "to": "return_bc", "linear_resistance": float(case.far_resistance),
             "initial_mass_flow": total},
        ],
        "ports": ports,
    }


def simulation_document(case: NetworkCase, conductor_files: list, network: dict | None) -> dict:
    document = {
        "format": "opensc2-yaml/1",
        "simulation": {
            "name": case.name,
            "end_time": float(case.end_time),
            "time_stepping": {
                "adaptivity": "FIXED",
                "minimum_step": float(case.time_step),
                "maximum_step": float(case.time_step),
                "reference_time": float(case.end_time),
                "reference_duration": float(case.end_time),
            },
        },
        "environment": {"medium": "Air", "temperature": 300.15, "pressure": 101325},
        "conductors": [{"file": name} for name in conductor_files],
    }
    if network is not None:
        document["hydraulic_network"] = network
    return document


def write_network_case(run_directory: Path, case: NetworkCase) -> Path:
    """Write the N-conductor + network run directory."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    template = load_template()
    files = []
    for spec in case.conductors:
        document = build_conductor_document(
            template, spec, case,
            mass_flow=case.mass_flows.get(spec.identifier, 1.0e-3),
            heated=bool(case.heated_strands.get(spec.identifier, False)),
        )
        name = f"conductor_{spec.identifier}.yaml"
        (run_directory / name).write_text(yaml.safe_dump(document, sort_keys=False))
        files.append(name)
    (run_directory / "simulation.yaml").write_text(
        yaml.safe_dump(simulation_document(case, files, network_section(case)), sort_keys=False)
    )
    return run_directory


def write_single_channel_case(run_directory: Path, length: float, elements: int,
                              time_step: float, end_time: float, formulation: str,
                              inlet_mass_flow: float, inlet_pressure: float,
                              outlet_pressure: float, temperature: float,
                              spatial_times: list, method: str = "BDF2",
                              name: str = "formulation_suite_tier4") -> Path:
    """Tier-4 deck: the template channel alone (thermally decoupled stub
    jacket, ``couplings: []``), imposed inlet mass flow and outlet pressure
    (hydraulic_boundary_condition 3)."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    template = load_template()
    spec = ConductorSpec(1, length, elements)
    case = NetworkCase([spec], time_step, end_time, formulation,
                       pressure_supply_volume=inlet_pressure,
                       pressure_return_volume=outlet_pressure, temperature=temperature,
                       spatial_times=spatial_times, method=method, name=name)
    document = build_conductor_document(template, spec, case, inlet_mass_flow, heated=False)
    channel = template_channel(document)
    channel["operations"]["hydraulic_boundary_condition"] = 3
    jacket = next(c for c in document["components"]
                  if c["kind"] == "Z_JACKET" and c["inputs"]["jacket_kind"] == "whole_enclosure")
    document["components"] = [channel, jacket]
    document["coupling_component_order"] = ["Environment", channel["identifier"], jacket["identifier"]]
    document["couplings"] = []
    (run_directory / "conductor_CONDUCTOR_1.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False)
    )
    (run_directory / "simulation.yaml").write_text(
        yaml.safe_dump(simulation_document(case, ["conductor_CONDUCTOR_1.yaml"], None),
                       sort_keys=False)
    )
    return run_directory
