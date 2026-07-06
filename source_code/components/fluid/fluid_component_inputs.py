"""
This module contains the dataclasses storing fluid component user inputs.
"""

from dataclasses import dataclass 
import pandas as pd

import interfaces.yaml_input_registry as yaml_input_registry 
from pathlib import Path 

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
        if self.initial_temperature_outlet is not None and not pd.isna(
            self.initial_temperature_outlet
        ):
            return self.initial_temperature, self.initial_temperature_outlet
        return self.inlet_temperature, self.outlet_temperature


class FluidComponentInputLoader:
    """Interface class used to get the input data for fluid components objects."""

    def __init__(self, input_file_path: str, operations_file_path: str,
                identifier):
        """Initializes the fluid component input loader.

        Parameters
        ----------
            input_file_path : str 
                path to the input file 
            operations_file_path: str
                path to the operations file 
            identifier : str 
                identifier of the fluid component 
        """
        # Set file paths 
        self.input_file = Path(input_file_path)
        self.operations_file = Path(operations_file_path)
        self.identifier = identifier


    def load_input_file(self) -> FluidComponentInputs:
        """Loads the fluid component data from the input file and returns a 
        FluidComponentInputs object with the data."""
        registry = yaml_input_registry.get_registry(self.input_file)
        if registry is not None:
            workbook = registry.component_raw_inputs(self.identifier)
        else:
            workbook = pd.read_excel(
                self.input_file,
                sheet_name="CHAN",
                skiprows=2,
                header=0,
                index_col=0,
                usecols=["Variable name", self.identifier],
            )[self.identifier].to_dict()
        return FluidComponentInputs(
            cross_section=workbook["CROSSECTION"],
            x_barycenter=workbook["X_barycenter"],
            y_barycenter=workbook["Y_barycenter"],
            cos_theta=workbook["COSTETA"],
            void_fraction=workbook["VOID_FRACTION"],
            hydraulic_diameter=workbook["HYDIAMETER"],
            roughness=workbook["Roughness"],
            is_rectangular=bool(workbook["ISRECTANGULAR"]),
            width=workbook["SIDE1"],
            height=workbook["SIDE2"],
            fluid_type=hf.get_fluid_type(workbook["FLUID_TYPE"]),
            friction_factor_model=FrictionFactorModelType.get_friction_factor_model(workbook["IFRICTION"]),
            heat_transfer_model=HeatTransferModelType.get_heat_transfer_model(workbook["Flag_htc_steady_corr"]),
            friction_multiplier=workbook["FRICTION_MULTIPLIER"],
            channel_type=hf.get_channel_type(workbook["CHANNEL_TYPE"]),
            show_figure=bool(workbook["Show_fig"])
        )
    

    def load_operations_file(self) -> FluidComponentOperations:
        registry = yaml_input_registry.get_registry(self.operations_file)
        if registry is not None:
            workbook = registry.component_raw_operations(self.identifier)
        else:
            workbook = pd.read_excel(
                self.operations_file,
                sheet_name="CHAN",
                skiprows=2,
                header=0,
                index_col=0,
                usecols=["Variable name", self.identifier],
            )[self.identifier].to_dict()
        return FluidComponentOperations(
            hydraulic_bc_type=hf.get_hydraulic_bc(int(workbook["INTIAL"])),
            bc_values_from_file=int(workbook["INTIAL"]) < 0,
            inlet_temperature=workbook["TEMINL"],
            outlet_temperature=workbook["TEMOUT"], 
            initial_temperature=workbook["TEMINI"],
            inlet_pressure=workbook["PREINL"],
            outlet_pressure=workbook["PREOUT"],
            initial_pressure=workbook["PREINI"],
            inlet_mass_rate=workbook["MDTIN"],
            outlet_mass_rate=workbook["MDTOUT"],
            flow_direction=hf.get_flow_direction(workbook["FLOWDIR"]),
            initial_temperature_outlet=workbook.get("TEMINI_OUT"),
        )