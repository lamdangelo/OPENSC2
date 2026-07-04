"""
Input dataclasses and loader for StackComponent.

TapeLayer               – one material layer in the HTS tape stack
StackComponentInputs    – all input parameters for a StackComponent
StackComponentInputLoader – reads Excel and constructs StackComponentInputs + StrandComponentOperations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import interfaces.yaml_input_registry as yaml_input_registry

from components.solid.solid_component_inputs import SolidComponentInputs, StrandComponentOperations
from utility_functions.auxiliary_functions import check_costheta
from conductor.conductor_flags import InterpolationType
from thermal.thermal_flags import HeatExcitation
from electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode


@dataclass
class TapeLayer:
    """One material layer in an HTS tape (e.g. HTS, buffer, substrate, stabilizer)."""
    name: str       # Excel key prefix, e.g. "HTS", "buffer", "substrate"
    material: str   # lowercase material identifier, e.g. "re123", "ag", "none"
    thickness: float  # layer thickness in m
    is_hts: bool = False  # True for the superconducting (HTS) layer


@dataclass
class StackComponentInputs(SolidComponentInputs):
    """Input parameters for StackComponent (HTS tape stack)."""

    # Superconducting material parameters (used in strand_component.py)
    superconducting_material: str          # superconducting_material
    upper_critical_field_at_0K: float      # Bc20m — Bc2 at T=0, B=0 in T
    critical_current_scaling_constant: float  # c0
    critical_temperature_at_0T: float      # Tc0m — Tc0 at B=0 in K

    # Stack geometry
    stack_width: float                     # Stack_width in m
    tape_identifier: str                   # Tape_number
    num_material_layers: int               # Material_number
    num_tapes: int                         # N_tape

    # Electrical parameters
    residual_resistivity_ratio: float      # RRR
    flux_flow_electric_field: float        # E0 in V/m
    power_law_exponent: float              # nn (n-value exponent)

    # Variable-length tape layers (all material-thickness pairs from Excel, in Excel row order)
    # HTS layer has is_hts=True; others are buffer/substrate/stabilizer etc.
    tape_layers: list[TapeLayer] = field(default_factory=list)


class StackComponentInputLoader:
    """Loads StackComponent inputs and operations from Excel files."""

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

    def load_input_file(self) -> StackComponentInputs:
        wb = self._read_sheet(self.input_file)

        # Collect tape layers: every key ending in "_material" has a paired key
        # ending in "_thickness" with the same prefix.
        tape_layers: list[TapeLayer] = []
        material_keys = [k for k in wb if k.endswith("material")]
        for mat_key in material_keys:
            prefix = mat_key[: -len("material")]  # e.g. "HTS_", "buffer_"
            thickness_key = prefix + "thickness"
            thickness = float(wb.get(thickness_key, 0.0))
            material = str(wb[mat_key]).lower()
            is_hts = mat_key == "HTS_material"
            layer_name = prefix.rstrip("_")
            tape_layers.append(TapeLayer(
                name=layer_name,
                material=material,
                thickness=thickness,
                is_hts=is_hts,
            ))

        inputs = StackComponentInputs(
            cross_section=float(wb["CROSSECTION"]),
            cos_theta=float(wb["COSTETA"]),
            x_barycenter=float(wb["X_barycenter"]),
            y_barycenter=float(wb["Y_barycenter"]),
            show_figure=bool(wb["Show_fig"]),
            superconducting_material=str(wb["superconducting_material"]).lower(),
            upper_critical_field_at_0K=float(wb["Bc20m"]),
            critical_current_scaling_constant=float(wb["c0"]),
            critical_temperature_at_0T=float(wb["Tc0m"]),
            stack_width=float(wb["Stack_width"]),
            tape_identifier=str(wb["Tape_number"]),
            num_material_layers=int(wb["Material_number"]),
            num_tapes=int(wb["N_tape"]),
            residual_resistivity_ratio=float(wb["RRR"]),
            flux_flow_electric_field=float(wb["E0"]),
            power_law_exponent=float(wb["nn"]),
            tape_layers=tape_layers,
        )

        # Validate cos_theta here (moved out of component constructor)
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
