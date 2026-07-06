"""
This module loads the Excel input files.
"""

from enum import Enum
from pathlib import Path
import numpy as np 
from typing import Any, Dict, List, Optional, TypeVar

import pandas as pd
from openpyxl import load_workbook

import conductor.conductor_flags as cf
import electromagnetics.electromagnetic_flags as emf

from conductor.conductor_inputs import (
    ConductorInputs, 
    ConductorInputFiles,
    ConductorOperations,
    ConductorCoupling
)
from conductor.conductor_mesh import ConductorMesh
from conductor.coupling import CouplingMatrix
import conductor.input_validator as validator
import conductor.external_inputs as external_inputs 
import interfaces.yaml_input_registry as yaml_input_registry

EnumType = TypeVar("EnumType", bound=Enum)


class ConductorInputLoader:
    """Load conductor-related input files from a directory."""

    def __init__(self, input_directory_path: str, conductor_counter: int):
        # Mandatory input file paths
        self.input_directory_path = Path(input_directory_path)
        self.definition_path = self.input_directory_path / "conductor_definition.xlsx"
        self.transitory_path = self._find_file_containing("transitory_input")
        self.environment_path = self._find_file_containing("environment")
        self.grid_path = self._find_file_containing("grid")
        self.diagnostics_path = self._find_file_containing("diagnostic")
        self.coupling_path = self._find_file_containing("coupling")

        # Optional input file paths 
        self.external_alphab_path = self._find_file_containing("external_alphab")
        self.external_bfield_path = self._find_file_containing("external_bfield")
        self.external_current_path = self._find_file_containing("external_current")
        self.external_flow_path = self._find_file_containing("external_flow")
        self.external_heat_path = self._find_file_containing("external_heat")
        self.external_strain_path = self._find_file_containing("external_strain")
        self.external_grid_path = self._find_file_containing("external_grid")
        self.external_contact_perimeter_path = self._find_file_containing("external_contact_perimeter")

        self.conductor_counter = conductor_counter
        self.yaml_registry = yaml_input_registry.get_registry(
            self.input_directory_path
        )
        if self.yaml_registry is not None:
            # YAML-driven directory: no definition workbook exists; the
            # registry serves every sheet read below.
            self.definition_path = (
                self.input_directory_path
                / yaml_input_registry.SIMULATION_FILE_NAME
            )
            self.list_conductor_sheets = None
            self.conductor_sheet_names = [
                "CONDUCTOR_files",
                "CONDUCTOR_input",
                "CONDUCTOR_operation",
                "CONDUCTOR_coupling",
            ]
            self.conductor_name = self.yaml_registry.conductor_name(
                self.conductor_counter
            )
            self.conductor_identifier = self.yaml_registry.conductor_identifier(
                self.conductor_counter
            )
        else:
            self.list_conductor_sheets = self.load_conductor_definition()
            self.conductor_sheet_names = self.load_conductor_sheet_names()
            self.conductor_name = self.list_conductor_sheets[0].cell(row=1, column=1).value
            self.conductor_identifier = self.list_conductor_sheets[0].cell(row=3, column=4 + self.conductor_counter).value


    def load_input_files(self) -> ConductorInputFiles:
        """Load all input paths and transient configuration from the input folder."""
        conductor_files = self.load_identifier_sheet(
            self.definition_path, self.conductor_sheet_names[0]
        )
        return ConductorInputFiles(
            base_path=self.input_directory_path,
            transitory_path=self.transitory_path,
            conductor_definition_path=self.definition_path,
            environment_path=self.environment_path,
            grid_path=self.grid_path,
            diagnostics_path=self.diagnostics_path,
            coupling_path=self.coupling_path,
            structure_elements_path=self.input_directory_path / conductor_files["STRUCTURE_ELEMENTS"],
            operation_path=self.input_directory_path / conductor_files["OPERATION"],
            external_alphab=self.external_alphab_path,
            external_bfield=self.external_bfield_path,
            external_contact_perimeter=self.external_contact_perimeter_path,
            external_current=self.external_current_path,
            external_flow=self.external_flow_path,
            external_grid=self.external_grid_path,
            external_heat=self.external_heat_path,
            external_strain=self.external_strain_path
        )


    def load_identifier_sheet(
        self, workbook_path: Path, sheet_name: str
    ) -> Dict[str, Any]:
        """Load a conductor definition sheet by identifier column."""
        if self.yaml_registry is not None:
            return {
                "CONDUCTOR_files": self.yaml_registry.conductor_files,
                "CONDUCTOR_input": self.yaml_registry.conductor_inputs,
                "CONDUCTOR_operation": self.yaml_registry.conductor_operations,
            }[sheet_name](self.conductor_counter)
        return pd.read_excel(
            workbook_path,
            sheet_name=sheet_name,
            skiprows=2,
            header=0,
            index_col=0,
            usecols=["Variable name", self.conductor_identifier],
        )[self.conductor_identifier].to_dict()


    def load_conductor_definition_dictionaries(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load the first three conductor definition sheets (files, inputs, operations)."""
        workbook_path = self.definition_path
        sheet_names = self.conductor_sheet_names[:3]
        if len(sheet_names) < 3:
            raise ValueError(
                "Expected at least three conductor definition sheet names."
            )
        return (
            self.load_identifier_sheet(workbook_path, sheet_names[0]),
            self.load_identifier_sheet(workbook_path, sheet_names[1]),
            self.load_identifier_sheet(workbook_path, sheet_names[2]),
        )
    

    def load_conductor_operations(self) -> ConductorOperations:
        """Load the conductor operation inputs and convert them to a ConductorOperations dataclass."""
        raw_data = self.load_identifier_sheet(self.definition_path, self.conductor_sheet_names[2])
        return self.build_conductor_operations(raw_data)


    def load_grid_input(self, conductor_length: float) -> ConductorMesh:
        """Load the conductor grid input dictionary for a single conductor identifier."""
        if self.yaml_registry is not None:
            return ConductorMesh(
                conductor_length,
                self.yaml_registry.grid_settings(self.conductor_counter),
            )
        grid_data =  pd.read_excel(
            self.grid_path,
            sheet_name="GRID",
            skiprows=2,
            header=0,
            index_col=0,
            usecols=["Variable name", self.conductor_identifier],
            dtype="object",
        )[self.conductor_identifier].to_dict()
        return ConductorMesh(conductor_length, grid_data)


    def load_coupling_data(self) -> ConductorCoupling:
        """Load all sheets from the conductor coupling workbook."""
        if self.yaml_registry is not None:
            workbook = self.yaml_registry.coupling_dataframes(
                self.conductor_counter
            )
        else:
            workbook = pd.read_excel(
                self.coupling_path,
                sheet_name=None,
                skiprows=1,
                header=0,
                index_col=0,
            )
        valid = validator.check_coupling_sheet_names(workbook.keys())
        if valid:
            return ConductorCoupling(
                    contact_perimeter_flag=CouplingMatrix(workbook["contact_perimeter_flag"], int),
                    contact_perimeter=CouplingMatrix(workbook["contact_perimeter"], float),
                    htc_choice=CouplingMatrix(workbook["HTC_choice"], cf.HTC_Choice),
                    htc_contact=CouplingMatrix(workbook["contact_HTC"], float),
                    thermal_contact_resistance=CouplingMatrix(workbook["thermal_contact_resistance"], float),
                    htc_multiplier=CouplingMatrix(workbook["HTC_multiplier"], float),
                    electric_conductance_mode=CouplingMatrix(workbook["electric_conductance_mode"], 
                                                            emf.ElectricConductanceMode),
                    electric_conductance=CouplingMatrix(workbook["electric_conductance"], float), 
                    open_perimeter_fraction=CouplingMatrix(workbook["open_perimeter_fract"], float), 
                    interface_thickness=CouplingMatrix(workbook["interf_thickness"], float),
                    transport_property_multiplier=CouplingMatrix(workbook["trans_transp_multiplier"], float),
                    view_factors=CouplingMatrix(workbook["view_factors"], float)
                )
        else:
            raise ValueError("Invalid coupling sheet names encountered.")


    def load_external_contact_perimeter(self) -> external_inputs.ExternalContactPerimeter:
        """Load and normalize variable contact perimeter sheets."""
        dict_df = pd.read_excel(
            self.external_contact_perimeter_path,
            sheet_name=None,
            header=0,
        )
        for df in dict_df.values():
            df.columns = [col_name.split(".")[0] for col_name in df.columns]
        return external_inputs.ExternalContactPerimeter( ... )


    def load_conductor_inputs(self) -> ConductorInputs:
        """Load conductor input values and convert them to a ConductorInputs dataclass."""
        raw_inputs = self.load_identifier_sheet(self.definition_path, 
                                                "CONDUCTOR_input")
        return self.build_conductor_inputs(raw_inputs)


    def build_conductor_inputs(self, raw_inputs: Dict[str, Any]) -> ConductorInputs:
        """Convert raw conductor input dictionary values into a ConductorInputs instance."""
        return ConductorInputs(
            zlength=raw_inputs["ZLENGTH"],
            diameter=raw_inputs["Diameter"],
            is_rectangular=bool(raw_inputs["Is_rectangular"]),
            width=raw_inputs["Width"],
            height=raw_inputs["Height"],
            is_joint=bool(raw_inputs["ISJOINT"]),
            current_mode=emf.CurrentMode.get_current_mode_flag(raw_inputs["I0_OP_MODE"]),
            initial_current=raw_inputs["I0_OP_TOT"],
            inlet_heated_zone_start=raw_inputs["XJBEG"],
            inlet_heated_zone_end=raw_inputs["XJBEIN"],
            outlet_heated_zone_start=raw_inputs["XJBEOUT"],
            outlet_heated_zone_end=raw_inputs["XJENOUT"],
            thermohydraulic_method=cf.MethodFlag.get_method_flag(raw_inputs["METHOD"]),
            upwind=bool(raw_inputs.get("UPWIND", False)),
            external_free_convection_correlation=cf.ExternalFreeConvectionCorrelation.get_external_free_convection_correlation_flag(
                raw_inputs["external_free_convection_correlation"]
            ),
            electric_method=cf.MethodFlag.get_method_flag(raw_inputs["ELECTRIC_METHOD"]),
            electric_time_step=raw_inputs["ELECTRIC_TIME_STEP"],
            phi_radiative=raw_inputs["Phi_rad"],
            phi_convective=raw_inputs["Phi_conv"],
        )
    

    def build_conductor_operations(self, raw_inputs: Dict[str, Any]) -> ConductorOperations:
        """Convert raw conductor operation dictionary values into a ConductorOperations instance."""
        return ConductorOperations(
            do_equipotential_surfaces_exist=bool(raw_inputs["EQUIPOTENTIAL_SURFACE_FLAG"]),
            number_of_equipotential_surfaces=raw_inputs["EQUIPOTENTIAL_SURFACE_NUMBER"],
            equipotential_surface_coordinates=self._convert_equipotential_surface_coordinate_to_array(
                raw_inputs["EQUIPOTENTIAL_SURFACE_COORDINATE"]
            ),
            maximum_iteration_number=raw_inputs["MAXIMUM_ITERATION_NUMBER"],
            inductance_mode=emf.InductanceMode.get_inductance_mode_flag(raw_inputs["INDUCTANCE_MODE"]),
            self_inductance_mode=emf.SelfInductanceMode.get_self_inductance_mode_flag(raw_inputs["SELF_INDUCTANCE_MODE"]),
            electric_solver=emf.ElectricSolver.get_electric_solver_flag(raw_inputs["ELECTRIC_SOLVER"]),
        )
    

    def _convert_equipotential_surface_coordinate_to_array(self, equipotential_coordinates: Any) -> np.ndarray:
        """Convert values corresponding to key EQUIPOTENTIAL_SURFACE_COORDINATE to numpy array 
        according to the original type (integer for single value or string for multiple values).

        Args:
            self (Self): conductor object.
        """
        if isinstance(equipotential_coordinates, int):
            return np.array([equipotential_coordinates], dtype=float)
        elif isinstance(equipotential_coordinates, str):
            return np.array(equipotential_coordinates.split(","), dtype=float)
        else:
            raise ValueError(
                f"Invalid type for EQUIPOTENTIAL_SURFACE_COORDINATE: {type(equipotential_coordinates)}."
            )
    

    def load_conductor_definition(self) -> List[Any]:
        """Load all conductor definition sheets."""
        conductor_definition_workbook = load_workbook(
            self.definition_path, data_only=True
        )
        list_conductor_sheets = [
            conductor_definition_workbook["CONDUCTOR_files"],
            conductor_definition_workbook["CONDUCTOR_input"],
            conductor_definition_workbook["CONDUCTOR_operation"],
        ]
        return list_conductor_sheets
    

    def load_conductor_sheet_names(self) -> List[str]:
        """Load the names of the conductor definition sheets."""
        conductor_definition_workbook = load_workbook(
            self.definition_path, data_only=True
        )
        return [sheet.title for sheet in conductor_definition_workbook.worksheets]


    def _find_file_containing(self, substring: str) -> Path:
        candidates = [
            path
            for path in self.input_directory_path.iterdir()
            if substring.lower() in path.name.lower()
        ]
        if not candidates:
            return None 
        elif len(candidates) > 1:
            raise FileExistsError(
                f"Multiple files containing '{substring}' were found in {self.input_directory_path}: {', '.join(str(p) for p in candidates)}"
            )
        else:
            return candidates[0]


    def _load_transient_input(self, starter_file: Path) -> Dict[str, Any]:
        return pd.read_excel(
            starter_file,
            sheet_name="TRANSIENT",
            skiprows=1,
            header=0,
            index_col=0,
            usecols=["Variable name", "Value"],
        )["Value"].to_dict()


    def _find_grid_and_diagnostic_paths(self):
        grid_path: Optional[Path] = None
        diagnostics_path: Optional[Path] = None

        for path in self.input_directory_path.iterdir():
            name = path.name.lower()
            if "grid" in name:
                grid_path = path
            elif "diagnostic" in name:
                diagnostics_path = path

        if grid_path is None:
            raise FileNotFoundError(
                f"Could not find a grid file in {self.input_directory_path}"
            )
        if diagnostics_path is None:
            raise FileNotFoundError(
                f"Could not find a diagnostic file in {self.input_directory_path}"
            )

        return grid_path, diagnostics_path
    

    def _find_coupling_path(self) -> Path:
        for path in self.input_directory_path.iterdir():
            if "coupling" in path.name.lower():
                return path
        raise FileNotFoundError(
            f"Could not find a coupling file in {self.input_directory_path}"
        )


    @staticmethod 
    def load_conductor_definition_sheets(conductor_definition_path: Path):
        conductor_definition_workbook = load_workbook(conductor_definition_path, data_only=True)
        list_conductor_sheets = [
            conductor_definition_workbook["CONDUCTOR_files"],
            conductor_definition_workbook["CONDUCTOR_input"],
            conductor_definition_workbook["CONDUCTOR_operation"],
        ]
        return list_conductor_sheets
