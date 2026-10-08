"""
This module contains the dataclasses storing fluid component user inputs.
"""

import math
from dataclasses import dataclass


import components.component_flags as cf
import hydraulics.hydraulic_flags as hf
from hydraulics.friction_factor_models import FrictionFactorModelType
from thermal.heat_transfer_models import HeatTransferModelType


@dataclass 
class FluidComponentInputs:
    """
    A class to store the input data from the input Excel file for the fluid component.
    """
    # Geometric data 
    cross_section: float  # m²
    x_barycenter: float  # m
    y_barycenter: float  # m
    cos_theta: float  # angle w.r.t. axis 
    void_fraction: float 
    hydraulic_diameter: float  # m
    roughness: float  # channel equivalent roughness in m
    is_rectangular: bool 
    width: float  # m
    height: float  # m

    # Fluid properties
    fluid_type: hf.FluidType
    friction_factor_model: FrictionFactorModelType
    heat_transfer_model: HeatTransferModelType
    friction_multiplier: float
    channel_type: hf.ChannelType

    # Post-processing-related data
    show_figure: bool 


@dataclass
class FluidComponentOperations:
    """
    A class to store the data from the operations Excel file for the fluid component.
    """
    hydraulic_bc_type: hf.HydraulicBC  # abs(INTIAL) — kind of boundary condition
    bc_values_from_file: bool  # sign of INTIAL — True if BC values come from the external flow file

    inlet_temperature: float  # K
    outlet_temperature: float  # K
    initial_temperature: float  # K

    inlet_pressure: float  # Pa
    outlet_pressure: float  # Pa
    initial_pressure: float  # Pa

    inlet_mass_rate: float  # kg/s
    outlet_mass_rate: float  # kg/s

    flow_direction: hf.FlowDirection  # backward or forward

    # Optional outlet value of the initial temperature profile (workbook row
    # TEMINI_OUT). When set, the initial temperature runs linearly from
    # initial_temperature (TEMINI) at the inlet to this value at the outlet,
    # decoupled from the boundary-condition temperatures — the analogue of
    # THEA's user-defined hydraulic initial condition. When absent, the
    # initial profile falls back to inlet_temperature/outlet_temperature.
    initial_temperature_outlet: float = None

    def initial_temperature_profile_bounds(self) -> tuple:
        """Inlet and outlet values of the initial temperature profile."""
        if self.initial_temperature_outlet is not None and not (
            isinstance(self.initial_temperature_outlet, float)
            and math.isnan(self.initial_temperature_outlet)
        ):
            return self.initial_temperature, self.initial_temperature_outlet
        return self.inlet_temperature, self.outlet_temperature
