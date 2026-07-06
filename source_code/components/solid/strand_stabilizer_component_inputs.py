"""
Input dataclasses and loader for StrandStabilizerComponent.

StrandStabilizerComponentInputs – all input parameters for a StrandStabilizerComponent
StrandStabilizerInputLoader      – reads Excel and constructs the dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

import interfaces.yaml_input_registry as yaml_input_registry

from components.solid.solid_component_inputs import SolidComponentInputs, StrandComponentOperations
from utility_functions.auxiliary_functions import check_costheta
from conductor.conductor_flags import InterpolationType
from thermal.thermal_flags import HeatExcitation
from electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode


@dataclass
class StrandStabilizerComponentInputs(SolidComponentInputs):
    """Input parameters for StrandStabilizerComponent (pure stabilizer strand)."""

    # Stabilizer material
    stabilizer_material: str                  # stabilizer_material (lowercased by loader)

    # Electrical parameter — Optional because it is only meaningful for Cu stabilizer;
    # the loader returns 0.0 if absent, and the component sets it to None if not Cu.
    residual_resistivity_ratio: Optional[float]  # RRR

    # Superconducting material parameters. A pure stabilizer strand has no
    # superconductor and its workbook sheet may omit these rows entirely; the
    # superconductor-property evaluation (electromagnetics/operating_conditions.py)
    # explicitly skips StrandStabilizerComponent, so they are never read then.
    superconducting_material: Optional[str] = None    # superconducting_material
    upper_critical_field_at_0K: Optional[float] = None         # Bc20m
    critical_current_scaling_constant: Optional[float] = None  # c0
    critical_temperature_at_0T: Optional[float] = None         # Tc0m


class StrandStabilizerInputLoader:
    """Loads StrandStabilizerComponent inputs and operations from Excel files."""

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

    def load_input_file(self) -> StrandStabilizerComponentInputs:
        wb = self._read_sheet(self.input_file)

        stabilizer_material = str(wb["stabilizer_material"]).lower()
        rrr = float(wb["RRR"]) if stabilizer_material == "cu" else None

        inputs = StrandStabilizerComponentInputs(
            cross_section=float(wb["CROSSECTION"]),
            cos_theta=float(wb["COSTETA"]),
            x_barycenter=float(wb["X_barycenter"]),
            y_barycenter=float(wb["Y_barycenter"]),
            show_figure=bool(wb["Show_fig"]),
            stabilizer_material=stabilizer_material,
            residual_resistivity_ratio=rrr,
            # Optional superconductor rows (see the dataclass note).
            superconducting_material=(
                str(wb["superconducting_material"]).lower()
                if "superconducting_material" in wb
                else None
            ),
            upper_critical_field_at_0K=(
                float(wb["Bc20m"]) if "Bc20m" in wb else None
            ),
            critical_current_scaling_constant=(
                float(wb["c0"]) if "c0" in wb else None
            ),
            critical_temperature_at_0T=(
                float(wb["Tc0m"]) if "Tc0m" in wb else None
            ),
        )

        check_costheta(inputs.cos_theta, str(self.input_file), self.sheet_name)

        return inputs

    def load_operations_file(self) -> StrandComponentOperations:
        wb = self._read_sheet(self.operations_file, registry_section="operations")

        iop_mode_raw = wb["IOP_MODE"]
        if isinstance(iop_mode_raw, bool):
            iop_mode_raw = 1 if iop_mode_raw else 0

        return StrandComponentOperations(
            magnetic_field_bc_mode=BFieldDefinitionType.get_bfield_definition_type(int(wb["IBIFUN"])),
            magnetic_field_units=str(wb.get("B_field_units", "")) or None,
            magnetic_field_inlet_initial=float(wb["BISS"]),
            magnetic_field_outlet_initial=float(wb["BOSS"]),
            magnetic_field_inlet_transient=float(wb["BITR"]),
            magnetic_field_outlet_transient=float(wb["BOTR"]),
            magnetic_field_interpolation=InterpolationType.get_interpolation_type(str(wb["B_INTERPOLATION"])),
            operating_current_mode=CurrentMode.get_current_mode_flag(iop_mode_raw),
            operating_current_interpolation=InterpolationType.get_interpolation_type(str(wb["IOP_INTERPOLATION"])),
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
            alpha_b_mode=int(wb["IALPHAB"]),
            alpha_b_interpolation=str(wb["ALPHAB_INTERPOLATION"]),
            strain_mode=int(wb["IEPS"]),
            strain_value=float(wb.get("EPS", 0.0)),
            tcs_evaluation=bool(wb.get("TCS_EVALUATION", False)),
            fix_potential_flag=bool(wb["FIX_POTENTIAL_FLAG"]),
            fix_potential_number=int(wb["FIX_POTENTIAL_NUMBER"]),
            fix_potential_coordinate=wb["FIX_POTENTIAL_COORDINATE"],
            fix_potential_value=wb["FIX_POTENTIAL_VALUE"],
        )
