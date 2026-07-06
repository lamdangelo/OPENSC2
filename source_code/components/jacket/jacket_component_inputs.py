"""
Input dataclasses and loader for JacketComponent.

JacketComponentInputs – all input parameters for a JacketComponent
JacketComponentInputLoader – reads Excel and constructs the dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import interfaces.yaml_input_registry as yaml_input_registry

from components.solid.solid_component_inputs import SolidComponentInputs, SolidComponentOperations
from thermal.thermal_flags import HeatExcitation
from electromagnetics.electromagnetic_flags import BFieldDefinitionType


@dataclass
class JacketComponentInputs(SolidComponentInputs):
    """Input parameters for JacketComponent (structural jacket + insulation)."""

    jacket_material: str              # jacket_material (lowercased by loader)
    insulation_material: str          # insulation_material (lowercased by loader)
    jacket_cross_section: float       # jacket_cross_section in m²
    insulation_cross_section: float   # insulation_cross_section in m²
    num_material_types: int           # NUM_MATERIAL_TYPES
    jacket_kind: str                  # Jacket_kind — e.g. "outer_insulation", "whole_enclosure"
    emissivity: float                 # Emissivity — surface emissivity for radiative heat transfer
    outer_perimeter: float            # Outer_perimeter — outer perimeter for radiation view factors
    inner_perimeter: float            # Inner_perimeter — inner perimeter for radiation view factors


class JacketComponentInputLoader:
    """Loads JacketComponent inputs and operations from Excel files."""

    def __init__(
        self,
        input_file_path: str | Path,
        operations_file_path: str | Path,
        identifier: str,
        sheet_name: str,
    ):
        self.input_file = Path(input_file_path)
        self.operations_file = Path(operations_file_path)
        self.identifier = identifier
        self.sheet_name = sheet_name

    def _read_sheet(self, file_path: Path, registry_section: str = "inputs") -> dict:
        registry = yaml_input_registry.get_registry(file_path)
        if registry is not None:
            # In YAML mode inputs and operations live in the same conductor
            # document, so the caller states which section it wants.
            if registry_section == "operations":
                return registry.component_raw_operations(self.identifier)
            return registry.component_raw_inputs(self.identifier)
        return pd.read_excel(
            file_path,
            sheet_name=self.sheet_name,
            skiprows=2,
            header=0,
            index_col=0,
            usecols=["Variable name", self.identifier],
        )[self.identifier].to_dict()

    @staticmethod
    def _material_name(raw_value) -> str:
        """Normalize a workbook material cell to a lowercase name.

        A cell holding the string "None" arrives as NaN from
        pandas.read_excel (which treats "None" as an NA marker); map it to
        the "none" placeholder the material tables expect.
        """
        name = str(raw_value).lower()
        return "none" if name == "nan" or raw_value is None else name

    def load_input_file(self) -> JacketComponentInputs:
        wb = self._read_sheet(self.input_file)

        jacket_xs = float(wb["jacket_cross_section"])
        insulation_xs = float(wb["insulation_cross_section"])

        return JacketComponentInputs(
            cross_section=jacket_xs + insulation_xs,
            cos_theta=float(wb.get("COSTETA", 1.0)),
            x_barycenter=float(wb.get("X_barycenter", 0.0)),
            y_barycenter=float(wb.get("Y_barycenter", 0.0)),
            show_figure=bool(wb.get("Show_fig", True)),
            jacket_material=self._material_name(wb["jacket_material"]),
            insulation_material=self._material_name(wb["insulation_material"]),
            jacket_cross_section=jacket_xs,
            insulation_cross_section=insulation_xs,
            num_material_types=int(wb["NUM_MATERIAL_TYPES"]),
            jacket_kind=str(wb.get("Jacket_kind", "")),
            emissivity=float(wb.get("Emissivity", 0.0)),
            outer_perimeter=float(wb.get("Outer_perimeter", 0.0)),
            inner_perimeter=float(wb.get("Inner_perimeter", 0.0)),
        )

    def load_operations_file(self) -> SolidComponentOperations:
        wb = self._read_sheet(self.operations_file, registry_section="operations")

        iop_mode_raw = wb["IOP_MODE"]
        if isinstance(iop_mode_raw, bool):
            iop_mode_raw = 1 if iop_mode_raw else 0

        return SolidComponentOperations(
            magnetic_field_bc_mode=BFieldDefinitionType.get_bfield_definition_type(int(wb["IBIFUN"])),
            magnetic_field_units=str(wb.get("B_field_units", "")) or None,
            magnetic_field_inlet_initial=float(wb["BISS"]),
            magnetic_field_outlet_initial=float(wb["BOSS"]),
            magnetic_field_inlet_transient=float(wb["BITR"]),
            magnetic_field_outlet_transient=float(wb["BOTR"]),
            magnetic_field_interpolation=str(wb["B_INTERPOLATION"]),
            operating_current_mode=iop_mode_raw,
            operating_current_interpolation=str(wb["IOP_INTERPOLATION"]),
            heat_flux_mode=HeatExcitation.get_heat_excitation(int(wb["IQFUN"])),
            heat_flux_time_start=float(wb["TQBEG"]),
            heat_flux_time_end=float(wb["TQEND"]),
            heat_flux_amplitude=float(wb["Q0"]),
            heat_flux_position_start=float(wb["XQBEG"]),
            heat_flux_position_end=float(wb["XQEND"]),
            heat_flux_interpolation=str(wb["Q_INTERPOLATION"]),
            initial_temperature_mode=int(wb["INTIAL"]),
            inlet_temperature=float(wb["TEMINL"]),
            outlet_temperature=float(wb["TEMOUT"]),
        )
