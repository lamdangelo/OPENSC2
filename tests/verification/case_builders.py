"""Builders shared by the verification tests and the figure scripts.

Network builders mirror the mapping-dict helpers of
tests/test_hydraulic_network.py, with the constant-property fluid instead of
helium so every case has exactly constant coefficients to compare against
closed-form solutions. The channel/coupled-run builders (V3/V4) emit a
minimal single-channel YAML run directory shaped like CASE_1 and drive
time-dependent boundary conditions through the production
Simulation(step_callback=...) hook.
"""

from pathlib import Path

import numpy as np
import yaml

import interfaces.coolprop_interface as cpi
from conductor.conductor_flags import MethodFlag
from hydraulics.network import HydraulicNetwork, HydraulicNetworkInput
from simulation import Simulation

# Default verification fluid: water-like magnitudes giving a moderate wave
# speed a = 1/sqrt(rho kappa_T) = 31.62 m/s, so acoustic transients are
# resolvable with CI-friendly time steps.
VERIFICATION_FLUID = cpi.ConstantFluidProperties(
    density=1000.0,  # kg/m^3
    isothermal_compressibility=1.0e-6,  # 1/Pa
    viscosity=1.0e-3,  # Pa s
    isobaric_specific_heat=4000.0,  # J/(kg K)
    thermal_conductivity=0.6,  # W/(m K)
)

BLASIUS_FLAG = 110  # smooth-tube laminar + Blasius turbulent, total = max
DEFAULT_TEMPERATURE = 300.0  # K

# Upper end of the exactly-laminar band of the Blasius friction pairing:
# 16/Re >= 0.0791 Re^-0.25 for Re <= (16/0.0791)^(4/3) ~ 1187, where the
# total = max(laminar, turbulent) model returns the laminar branch and the
# pipe resistance is exactly Hagen-Poiseuille.
LAMINAR_REYNOLDS_LIMIT = (16.0 / 0.0791) ** (4.0 / 3.0)


# --------------------------------------------------------------------- #
# Network mapping builders                                              #
# --------------------------------------------------------------------- #


def reservoir(identifier, pressure, temperature=DEFAULT_TEMPERATURE):
    return {
        "identifier": identifier,
        "kind": "reservoir",
        "pressure": pressure,
        "temperature": temperature,
    }


def internal(identifier, pressure, temperature=DEFAULT_TEMPERATURE, volume=0.0):
    return {
        "identifier": identifier,
        "kind": "internal",
        "initial_pressure": pressure,
        "initial_temperature": temperature,
        "volume": volume,
    }


def pipe(identifier, from_node, to_node, length, diameter, cross_section, **extra):
    return {
        "identifier": identifier,
        "kind": "pipe",
        "from": from_node,
        "to": to_node,
        "length": length,
        "hydraulic_diameter": diameter,
        "cross_section": cross_section,
        "friction_factor_model": BLASIUS_FLAG,
        **extra,
    }


def valve(identifier, from_node, to_node, **extra):
    return {
        "identifier": identifier,
        "kind": "valve",
        "from": from_node,
        "to": to_node,
        **extra,
    }


def network_mapping(nodes, branches, ports=None):
    mapping = {"fluid_type": "constant", "nodes": nodes, "branches": branches}
    if ports is not None:
        mapping["ports"] = ports
    return mapping


def build_network(nodes, branches, ports=None, method=MethodFlag.BACKWARD_EULER):
    return HydraulicNetwork(
        HydraulicNetworkInput.from_mapping(network_mapping(nodes, branches, ports)),
        method=method,
    )


# --------------------------------------------------------------------- #
# Single-channel run directory (V3/V4)                                  #
# --------------------------------------------------------------------- #

CHANNEL_LENGTH = 10.0  # m
CHANNEL_DIAMETER = 1.0e-2  # m
CHANNEL_CROSS_SECTION = np.pi / 4.0 * CHANNEL_DIAMETER**2  # m^2

# Fluid for the wave-dynamics cases (V3/V4): identical to VERIFICATION_FLUID
# except for a 100x lower viscosity. The laminar friction drop over the line
# scales like R mdot0 and the Joukowsky rise like (a/A) mdot0, so their
# ratio 32 mu L / (D^2 rho a) is independent of the flow; at mu = 1e-3 it is
# 10% (line packing visibly tilts the water-hammer plateau), at mu = 1e-5 it
# is 0.1% -- negligible against the 1-2% wave-dynamics tolerances.
ACOUSTIC_FLUID = cpi.ConstantFluidProperties(
    density=VERIFICATION_FLUID.density,
    isothermal_compressibility=VERIFICATION_FLUID.isothermal_compressibility,
    viscosity=1.0e-5,  # Pa s
    isobaric_specific_heat=VERIFICATION_FLUID.isobaric_specific_heat,
    thermal_conductivity=VERIFICATION_FLUID.thermal_conductivity,
)

WAVE_SPEED = 1.0 / np.sqrt(
    ACOUSTIC_FLUID.density * ACOUSTIC_FLUID.isothermal_compressibility
)  # m/s
# Characteristic impedance in (p, mdot) variables: Z = a/A (see
# analytical.characteristic_impedance for the derivation; the density
# cancels, and dimensionally a/A = 1/(m s) = Pa/(kg/s)).
CHARACTERISTIC_IMPEDANCE = WAVE_SPEED / CHANNEL_CROSS_SECTION  # Pa/(kg/s)


def write_channel_run_directory(
    run_directory,
    *,
    number_of_elements,
    time_step,
    end_time,
    method="BDF2",
    hydraulic_boundary_condition=2,
    inlet_pressure=1.0e5,
    outlet_pressure=1.0e5,
    initial_pressure=1.0e5,
    inlet_mass_rate=0.0,
    outlet_mass_rate=0.0,
    temperature=DEFAULT_TEMPERATURE,
    length=CHANNEL_LENGTH,
    network_section=None,
):
    """Write a minimal single-channel YAML run directory (CASE_1-shaped) for
    the constant-property fluid: one CHAN component, no solids, no couplings,
    no electric problem, fixed time stepping. Returns the directory path."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)

    # yaml.safe_dump cannot represent numpy scalars.
    time_step = float(time_step)
    end_time = float(end_time)
    inlet_pressure = float(inlet_pressure)
    outlet_pressure = float(outlet_pressure)
    initial_pressure = float(initial_pressure)
    inlet_mass_rate = float(inlet_mass_rate)
    outlet_mass_rate = float(outlet_mass_rate)
    temperature = float(temperature)
    length = float(length)

    simulation_document = {
        "format": "opensc2-yaml/1",
        "simulation": {
            "name": "verification",
            "end_time": end_time,
            "time_stepping": {
                # FIXED adaptivity forces conductor.time_step = minimum_step
                # every step (only the final step is truncated to land on
                # end_time).
                "adaptivity": "FIXED",
                "minimum_step": time_step,
                "maximum_step": time_step,
                "reference_time": end_time,
                "reference_duration": end_time,
            },
        },
        "environment": {
            "medium": "Air",
            "temperature": 300.15,
            "pressure": 101325,
        },
        "conductors": [{"file": "conductor_CONDUCTOR_1.yaml"}],
    }
    if network_section is not None:
        simulation_document["hydraulic_network"] = network_section

    conductor_document = {
        "conductor": {
            "name": "CONDUCTOR",
            "identifier": "CONDUCTOR_1",
            "inputs": {
                "length": length,
                "diameter": 0.0432,
                "is_rectangular": False,
                "width": 0,
                "height": 0,
                "current_mode": "none",
                "initial_current": 0,
                "is_joint": 0,
                "inlet_heated_zone_start": 0,
                "inlet_heated_zone_end": 0,
                "outlet_heated_zone_start": 0,
                "outlet_heated_zone_end": 0,
                "thermohydraulic_method": method,
                "upwind": 1,
                "external_free_convection_correlation": (
                    "vertical_plate_churchill_chu_accurate"
                ),
                "phi_radiative": 0.5,
                "phi_convective": 0.5,
                "electric_method": "BE",
                "electric_time_step": "None",
                "hydraulic_formulation": "mass_flow",
            },
            "operations": {
                "do_equipotential_surfaces_exist": True,
                "number_of_equipotential_surfaces": 1,
                "equipotential_surface_coordinates": 1,
                "maximum_iteration_number": 1000,
                "inductance_mode": 1,
                "self_inductance_mode": 2,
                "electric_solver": 1,
            },
        },
        "components": [
            {
                "identifier": "CHAN_1",
                "kind": "CHAN",
                "sheet": "CHAN",
                "inputs": {
                    "cross_section": float(CHANNEL_CROSS_SECTION),
                    "x_barycenter": 0,
                    "y_barycenter": 0,
                    "fluid_type": "constant",
                    "hydraulic_diameter": CHANNEL_DIAMETER,
                    "roughness": 0,
                    "cos_theta": 1,
                    "void_fraction": 1,
                    "friction_factor_model": BLASIUS_FLAG,
                    "friction_multiplier": 1,
                    "is_rectangular": False,
                    "width": 0,
                    "height": 0,
                    "channel_type": "hole",
                    "heat_transfer_model": 1,
                    "show_figure": False,
                },
                "operations": {
                    "hydraulic_boundary_condition": hydraulic_boundary_condition,
                    "boundary_values_from_file": False,
                    "inlet_temperature": temperature,
                    "outlet_temperature": temperature,
                    "initial_temperature": temperature,
                    "inlet_pressure": inlet_pressure,
                    "outlet_pressure": outlet_pressure,
                    "initial_pressure": initial_pressure,
                    "inlet_mass_flow_rate": inlet_mass_rate,
                    "outlet_mass_flow_rate": outlet_mass_rate,
                    "flow_direction": "forward",
                },
            },
            {
                # Stub jacket, thermally decoupled (couplings list is empty,
                # so no interface exists and the channel hydraulics are
                # untouched). It is only here because the post-processing
                # plot generator assumes at least one jacket per conductor
                # (plots.py make_plots, kind="Space_distr", reuses the
                # jacket loop variable after the loop).
                "identifier": "Z_JACKET_1",
                "kind": "Z_JACKET",
                "sheet": "Z_JACKET",
                "inputs": {
                    "jacket_cross_section": 0.00030699,
                    "insulation_cross_section": 0.00027966,
                    "x_barycenter": 0,
                    "y_barycenter": 0,
                    "inner_perimeter": 0.1288,
                    "outer_perimeter": 0.1357,
                    "number_of_material_types": 2,
                    "jacket_material": "ss",
                    "insulation_material": "ge",
                    "cos_theta": 1,
                    "jacket_kind": "whole_enclosure",
                    "emissivity": 1,
                    "show_figure": False,
                },
                "operations": {
                    "operating_current_mode": 0,
                    "operating_current_interpolation": "linear",
                    "magnetic_field_mode": 0,
                    "magnetic_field_interpolation": "linear",
                    "magnetic_field_units": "T",
                    "magnetic_field_inlet_initial": 0,
                    "magnetic_field_outlet_initial": 0,
                    "magnetic_field_inlet_transient": 0,
                    "magnetic_field_outlet_transient": 0,
                    "fixed_field_angle_value": 0,
                    "field_angle_mode": 0,
                    "field_angle_interpolation": "linear",
                    "heat_flux_mode": 0,
                    "heat_flux_interpolation": "linear",
                    "heat_flux_position_start": 0,
                    "heat_flux_position_end": 0,
                    "heat_flux_amplitude": 0,
                    "heat_flux_time_start": 0,
                    "heat_flux_time_end": 0,
                    "initial_temperature_mode": 0,
                    "inlet_temperature": 0,
                    "outlet_temperature": 0,
                },
            },
        ],
        "coupling_component_order": ["Environment", "CHAN_1", "Z_JACKET_1"],
        "couplings": [],
        "grid": {
            "number_of_elements": number_of_elements,
            "mesh_type": 0,
            "refined_zone_number_of_elements": 0,
            "refined_zone_start": 0,
            "refined_zone_end": 0,
            "minimum_element_size": 0.5,
            "maximum_element_size": 2,
            "maximum_number_of_nodes": 10001,
            "growth_ratio_left": 1.2,
            "growth_ratio_right": 1.2,
        },
        "diagnostics": {
            "spatial_distribution_times": [end_time],
            "time_evolution_positions": [length / 2.0],
        },
    }

    (run_directory / "simulation.yaml").write_text(
        yaml.safe_dump(simulation_document, sort_keys=False)
    )
    (run_directory / "conductor_CONDUCTOR_1.yaml").write_text(
        yaml.safe_dump(conductor_document, sort_keys=False)
    )
    return run_directory


class ChannelProbe:
    """Step callback recording the channel state at one mesh node.

    The hook fires at the top of every transient iteration, i.e. it sees the
    state of the previously completed step at simulation_time[-1]; the final
    state must be appended by calling record_final(simulation) after run().
    """

    def __init__(self, node_index, channel_identifier="CHAN_1"):
        self.node_index = node_index
        self.channel_identifier = channel_identifier
        self.times = []
        self.pressures = []
        self.mass_flow_rates = []

    def _channel(self, simulation):
        conductor = simulation.list_of_Conductors[0]
        return next(
            component
            for component in conductor.inventory.fluids.collection
            if component.identifier == self.channel_identifier
        )

    def record(self, simulation):
        channel = self._channel(simulation)
        self.times.append(simulation.simulation_time[-1])
        self.pressures.append(
            float(channel.coolant.node_fields.pressure[self.node_index])
        )
        self.mass_flow_rates.append(
            float(channel.coolant.node_fields.mass_flow_rate[self.node_index])
        )

    record_final = record

    def __call__(self, simulation):
        self.record(simulation)

    def as_arrays(self):
        return (
            np.asarray(self.times),
            np.asarray(self.pressures),
            np.asarray(self.mass_flow_rates),
        )


class StepDriver:
    """Compose per-step actions (probes, boundary schedules) into the single
    Simulation step_callback. Actions are called in order with the
    simulation; their return values are ignored (the transient never stops
    early)."""

    def __init__(self, *actions):
        self.actions = actions

    def __call__(self, simulation):
        for action in self.actions:
            action(simulation)
        return False


def channel_operations(simulation, channel_identifier="CHAN_1"):
    """The mutable FluidComponentOperations of a channel: boundary values are
    re-read from it at every assembly, so mutating it inside a step callback
    imposes time-dependent boundary conditions through the production path."""
    conductor = simulation.list_of_Conductors[0]
    channel = next(
        component
        for component in conductor.inventory.fluids.collection
        if component.identifier == channel_identifier
    )
    return channel.coolant.operations


def run_case(run_directory, step_callback=None):
    """Run the production Simulation on a prepared run directory."""
    simulation = Simulation(str(run_directory), step_callback=step_callback)
    simulation.run()
    return simulation
