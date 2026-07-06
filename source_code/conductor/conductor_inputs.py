"""
This module contains various data classes which store the input data from the input Excel files.
"""

from dataclasses import dataclass
import numpy as np 
from pathlib import Path

from conductor.conductor_flags import (
    MethodFlag,
    ExternalFreeConvectionCorrelation,
    HTC_Choice,
)
from electromagnetics.electromagnetic_flags import (
    CurrentMode,
    InductanceMode,
    SelfInductanceMode,
    ElectricSolver,
    ElectricConductanceMode,
)
from conductor.coupling import CouplingMatrix


@dataclass
class ConductorInputs:
    """
    A class to store the input data from the input Excel files for the conductor simulation.
    """
    # Geometric data 
    zlength: float  # length of the conductor 
    diameter: float  # diameter of the conductor if it is cylindrical
    is_rectangular: bool  # indicator whether the conductor has a rectangular cross-section
    width: float  # width of the conductor if it is rectangular
    height: float   # height of the conductor if it is rectangular
    is_joint: bool  # indicator whether the conductor has a joint

    # Current data 
    current_mode: CurrentMode  # I0_OP_MODE
    initial_current: float  # I0_OP_TOT - initial transport current 

    # Heated zone data in the inlet/outlet joints, if any
    inlet_heated_zone_start: float  # XJBEG - start of the heated zone in the inlet joint
    inlet_heated_zone_end: float  # XJBEIN - end of the heated zone in the inlet joint
    outlet_heated_zone_start: float  # XJBEOUT - start of the heated zone in the outlet joint
    outlet_heated_zone_end: float  # XJENOUT - end of the heated zone in the outlet joint

    # Solver data 
    thermohydraulic_method: MethodFlag # METHOD - numerical method for the thermohydraulic problem
    upwind: bool # UPWIND - whether to use upwind scheme for the spatial discretization of the thermohydraulic problem
    external_free_convection_correlation: ExternalFreeConvectionCorrelation  
    electric_method: MethodFlag  # ELECTRIC_METHOD - numerical method for the electric problem
    electric_time_step: float 

    # Fractions of outer lateral surfaces of the conductor subjected to heat exchange 
    phi_radiative: float  # Phi_rad - subjected to radiative heat exchange 
    phi_convective: float  # Phi_conv - subjected to convective heat exchange 


@dataclass
class ConductorInputFiles:
    """
    A class to store the paths to the input files for the conductor simulation.
    """
    # Mandatory input files
    base_path: Path
    transitory_path: Path
    conductor_definition_path: Path
    environment_path: Path
    grid_path: Path
    diagnostics_path: Path
    coupling_path: Path
    structure_elements_path: Path  # STRUCTURE_ELEMENTS - per-conductor component definition workbook
    operation_path: Path  # OPERATION - per-conductor component operation workbook

    # Optional input files 
    external_alphab: Path 
    external_bfield: Path 
    external_current: Path 
    external_flow: Path 
    external_heat: Path 
    external_strain: Path 
    external_grid: Path 
    external_contact_perimeter: Path


    def external_contact_perimeter_exists(self) -> bool:
        return self.external_contact_perimeter is not None 


@dataclass 
class ConductorOperations:
    """
    A class to store the operation parameters for the conductor simulation.
    """
    do_equipotential_surfaces_exist: bool  # EQUIPOTENTIAL_SURFACE_FLAG : whether equipotential surfaces exist 
    number_of_equipotential_surfaces: int  # EQUIPOTENTIAL_SURFACE_NUMBER : number of equipotential surfaces if they exist
    equipotential_surface_coordinates: np.ndarray  # EQUIPOTENTIAL_SURFACE_COORDINATE : list of axial coordinates 
    maximum_iteration_number: int  # MAXIMUM_ITERATION_NUMBER for the iterative methods in the electric solver
    inductance_mode: InductanceMode  # INDUCTANCE_MODE : method to evaluate the indutance
    self_inductance_mode: SelfInductanceMode  # SELF_INDUCTANCE_MODE : method to evaluate the self-inductance
    electric_solver: ElectricSolver  # ELECTRIC_SOLVER : solver for the electric problem (steady-state or transient)


@dataclass
class ConductorCoupling:
    """
    A class to store the coupling data between conductor components.
    """
    contact_perimeter_flag: CouplingMatrix[int]  # whether the components are in contact (values of ContactPerimeterFlag)
    contact_perimeter: CouplingMatrix[float]  # contact perimeter between components in m
    htc_choice: CouplingMatrix[HTC_Choice]  # choice of the heat transfer coefficient 
    htc_contact: CouplingMatrix[float]  # heat transfer coefficient in W/m²K
    thermal_contact_resistance: CouplingMatrix[float]  # thermal contact resistance in m²K/W
    htc_multiplier: CouplingMatrix[float]  # value for the multiplier for the HTC when needed
    electric_conductance_mode: CouplingMatrix[ElectricConductanceMode]  # kind of conductance 
    electric_conductance: CouplingMatrix[float]  # electric conductance in S/m
    open_perimeter_fraction: CouplingMatrix[float]  # fraction of the contact perimeter between fluid elements
    interface_thickness: CouplingMatrix[float]  # thickness of the interface between conductor components in m
    transport_property_multiplier: CouplingMatrix[float]  # multiplier for the transport properties 
    view_factors: CouplingMatrix[float]  # view factors between jackets to evaluate radiative heating


    def is_physically_valid(self) -> bool:
        return not any(
            matrix.check_for_negative_data()
            for matrix in (
                self.contact_perimeter,
                self.htc_contact,
                self.thermal_contact_resistance,
                self.electric_conductance,
                self.open_perimeter_fraction,
                self.interface_thickness
            )
        )
