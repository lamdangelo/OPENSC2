"""Builders of the component input/operation dataclasses from the raw sheet
dictionaries served by an InputSource (ports.input_source).

Format-agnostic: the YAML and Excel sources deliver the same legacy-keyed
dictionaries, so one builder per component kind suffices.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import opensc2.components.component_flags as cf
import opensc2.hydraulics.hydraulic_flags as hf
from opensc2.components.component_ordering import check_costheta
from opensc2.components.fluid.fluid_component_inputs import (
    FluidComponentInputs,
    FluidComponentOperations,
)
from opensc2.components.jacket.jacket_component_inputs import JacketComponentInputs
from opensc2.components.solid.solid_component_inputs import (
    SolidComponentInputs,
    SolidComponentOperations,
    StrandComponentOperations,
    finalize_solid_operations,
    loss_constant,
)
from opensc2.components.solid.stack_component_inputs import StackComponentInputs, TapeLayer
from opensc2.components.solid.strand_mixed_component_inputs import StrandMixedComponentInputs
from opensc2.components.solid.strand_stabilizer_component_inputs import (
    StrandStabilizerComponentInputs,
)
from opensc2.conductor.conductor_flags import InterpolationType
from opensc2.electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode
from opensc2.hydraulics.friction_factor_models import FrictionFactorModelType
from opensc2.ports.input_source import InputSource
from opensc2.thermal.heat_transfer_models import HeatTransferModelType
from opensc2.thermal.thermal_flags import HeatExcitation


class FluidComponentInputLoader:
    """Interface class used to get the input data for fluid components objects."""

    def __init__(self, source: InputSource, conductor_counter: int,
                 sheet_name: str, identifier: str):
        self.source = source
        self.conductor_counter = conductor_counter
        self.sheet_name = sheet_name
        self.identifier = identifier


    def load_input_file(self) -> FluidComponentInputs:
        """Loads the fluid component data from the input file and returns a 
        FluidComponentInputs object with the data."""
        workbook = self.source.component_raw_inputs(
            self.conductor_counter, self.sheet_name, self.identifier
        )
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
        workbook = self.source.component_raw_operations(
            self.conductor_counter, self.sheet_name, self.identifier
        )
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


class JacketComponentInputLoader:
    """Loads JacketComponent inputs and operations from Excel files."""

    def __init__(self, source: InputSource, conductor_counter: int,
                 identifier: str, sheet_name: str):
        self.source = source
        self.conductor_counter = conductor_counter
        self.identifier = identifier
        self.sheet_name = sheet_name
        # Informational only (error messages).
        self.input_file = f"{sheet_name} inputs"
        self.operations_file = f"{sheet_name} operations"

    def _read_sheet(self, file_path, registry_section: str = "inputs") -> dict:
        if registry_section == "operations":
            return self.source.component_raw_operations(
                self.conductor_counter, self.sheet_name, self.identifier
            )
        return self.source.component_raw_inputs(
            self.conductor_counter, self.sheet_name, self.identifier
        )

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

        return finalize_solid_operations(SolidComponentOperations(
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
            transverse_coupling_file=str(
                wb.get("TRANSVERSE_COUPLING_FILE", "") or ""
            ),
            eddy_loss_geometry_constant=float(
                wb.get("EDDY_LOSS_GEOMETRY_CONSTANT", 0.0) or 0.0
            ),
        ), self.identifier)


class StackComponentInputLoader:
    """Loads StackComponent inputs and operations from Excel files."""

    def __init__(self, source: InputSource, conductor_counter: int,
                 identifier: str, sheet_name: str):
        self.source = source
        self.conductor_counter = conductor_counter
        self.identifier = identifier
        self.sheet_name = sheet_name
        # Informational only (error messages).
        self.input_file = f"{sheet_name} inputs"
        self.operations_file = f"{sheet_name} operations"

    def _read_sheet(self, file_path, registry_section: str = "inputs") -> dict:
        if registry_section == "operations":
            return self.source.component_raw_operations(
                self.conductor_counter, self.sheet_name, self.identifier
            )
        return self.source.component_raw_inputs(
            self.conductor_counter, self.sheet_name, self.identifier
        )

    def load_input_file(self) -> StackComponentInputs:
        wb = self._read_sheet(self.input_file)

        # Collect tape layers: every key ending in "_material" has a paired key
        # ending in "_thickness" with the same prefix. The bare
        # "superconducting_material" key is the Jc-model selector, not a
        # tape layer, and has no paired thickness.
        tape_layers: list[TapeLayer] = []
        material_keys = [
            k for k in wb
            if k.endswith("material") and k != "superconducting_material"
        ]
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

        return finalize_solid_operations(StrandComponentOperations(
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
            # Optional AC-loss / proportional-field keys (same handling
            # as StrandMixedComponent; silently defaulting them dropped
            # the deck values for stacks - seen 2026-09-05, HELIAS 2D).
            magnetic_field_scales_with_current=bool(
                wb.get("B_SCALES_WITH_CURRENT", False)
            ),
            coupling_loss_time_constant=loss_constant(
                wb.get("COUPLING_LOSS_TIME_CONSTANT", 0.0)
            ),
            coupling_loss_relaxation_time=loss_constant(
                wb.get("COUPLING_LOSS_RELAXATION_TIME", 0.0)
            ),
            coupling_loss_copper_scaling=bool(
                wb.get("COUPLING_LOSS_COPPER_SCALING", False) or False
            ),
            coupling_loss_reference_temperature=float(
                wb.get("COUPLING_LOSS_REFERENCE_TEMPERATURE", 0.0) or 0.0
            ),
            coupling_loss_reference_field=float(
                wb.get("COUPLING_LOSS_REFERENCE_FIELD", 0.0) or 0.0
            ),
            tape_hysteresis_loss=bool(
                wb.get("TAPE_HYSTERESIS_LOSS", False) or False
            ),
            eddy_loss_geometry_constant=float(
                wb.get("EDDY_LOSS_GEOMETRY_CONSTANT", 0.0) or 0.0
            ),
            filament_diameter=float(
                wb.get("FILAMENT_DIAMETER", 0.0) or 0.0
            ),
            transverse_coupling_file=str(
                wb.get("TRANSVERSE_COUPLING_FILE", "") or ""
            ),
        ), self.identifier)


class StrandMixedInputLoader:
    """Loads StrandMixedComponent inputs and operations from Excel files."""

    def __init__(self, source: InputSource, conductor_counter: int,
                 identifier: str, sheet_name: str):
        self.source = source
        self.conductor_counter = conductor_counter
        self.identifier = identifier
        self.sheet_name = sheet_name
        # Informational only (error messages).
        self.input_file = f"{sheet_name} inputs"
        self.operations_file = f"{sheet_name} operations"

    def _read_sheet(self, file_path, registry_section: str = "inputs") -> dict:
        if registry_section == "operations":
            return self.source.component_raw_operations(
                self.conductor_counter, self.sheet_name, self.identifier
            )
        return self.source.component_raw_inputs(
            self.conductor_counter, self.sheet_name, self.identifier
        )

    def load_input_file(self) -> StrandMixedComponentInputs:
        wb = self._read_sheet(self.input_file)

        inputs = StrandMixedComponentInputs(
            cross_section=float(wb["CROSSECTION"]),
            cos_theta=float(wb["COSTETA"]),
            x_barycenter=float(wb["X_barycenter"]),
            y_barycenter=float(wb["Y_barycenter"]),
            show_figure=bool(wb["Show_fig"]),
            superconducting_material=str(wb["superconducting_material"]).lower(),
            upper_critical_field_at_0K=float(wb["Bc20m"]),
            critical_current_scaling_constant=float(wb["c0"]),
            critical_temperature_at_0T=float(wb["Tc0m"]),
            stabilizer_material=str(wb["stabilizer_material"]).lower(),
            num_sc_strands=int(wb["N_sc_strand"]),
            sc_strand_diameter=float(wb["d_sc_strand"]),
            num_stabilizer_strands=int(wb["N_stab_strand"]),
            stabilizer_strand_diameter=float(wb["d_stab_strand"]),
            stabilizer_to_sc_ratio_mode=int(wb["ISTAB_NON_STAB"]),
            stabilizer_to_sc_ratio=float(wb["STAB_NON_STAB"]),
            critical_current_definition_mode=int(wb["C0_MODE"]),
            num_material_types=int(wb["NUM_MATERIAL_TYPES"]),
            flux_flow_electric_field=float(wb["E0"]),
            power_law_exponent=float(wb["nn"]),
            residual_resistivity_ratio=float(wb.get("RRR", 0.0)),
        )

        check_costheta(inputs.cos_theta, str(self.input_file), self.sheet_name)

        return inputs

    def load_operations_file(self) -> StrandComponentOperations:
        wb = self._read_sheet(self.operations_file, registry_section="operations")

        iop_mode_raw = wb["IOP_MODE"]
        if isinstance(iop_mode_raw, bool):
            iop_mode_raw = 1 if iop_mode_raw else 0

        return finalize_solid_operations(StrandComponentOperations(
            magnetic_field_bc_mode=BFieldDefinitionType.get_bfield_definition_type(int(wb["IBIFUN"])),
            magnetic_field_units=str(wb.get("B_field_units", "")) or None,
            magnetic_field_inlet_initial=float(wb["BISS"]),
            magnetic_field_outlet_initial=float(wb["BOSS"]),
            magnetic_field_inlet_transient=float(wb["BITR"]),
            magnetic_field_outlet_transient=float(wb["BOTR"]),
            magnetic_field_interpolation=InterpolationType.get_interpolation_type(str(wb["B_INTERPOLATION"])),
            magnetic_field_scales_with_current=bool(wb.get("B_SCALES_WITH_CURRENT", False)),
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
            coupling_loss_time_constant=loss_constant(
                wb.get("COUPLING_LOSS_TIME_CONSTANT", 0.0)
            ),
            coupling_loss_relaxation_time=loss_constant(
                wb.get("COUPLING_LOSS_RELAXATION_TIME", 0.0)
            ),
            coupling_loss_copper_scaling=bool(
                wb.get("COUPLING_LOSS_COPPER_SCALING", False) or False
            ),
            coupling_loss_reference_temperature=float(
                wb.get("COUPLING_LOSS_REFERENCE_TEMPERATURE", 0.0) or 0.0
            ),
            coupling_loss_reference_field=float(
                wb.get("COUPLING_LOSS_REFERENCE_FIELD", 0.0) or 0.0
            ),
            tape_hysteresis_loss=bool(
                wb.get("TAPE_HYSTERESIS_LOSS", False) or False
            ),
            eddy_loss_geometry_constant=float(
                wb.get("EDDY_LOSS_GEOMETRY_CONSTANT", 0.0) or 0.0
            ),
            filament_diameter=float(
                wb.get("FILAMENT_DIAMETER", 0.0) or 0.0
            ),
        ), self.identifier)


class StrandStabilizerInputLoader:
    """Loads StrandStabilizerComponent inputs and operations from Excel files."""

    def __init__(self, source: InputSource, conductor_counter: int,
                 identifier: str, sheet_name: str):
        self.source = source
        self.conductor_counter = conductor_counter
        self.identifier = identifier
        self.sheet_name = sheet_name
        # Informational only (error messages).
        self.input_file = f"{sheet_name} inputs"
        self.operations_file = f"{sheet_name} operations"

    def _read_sheet(self, file_path, registry_section: str = "inputs") -> dict:
        if registry_section == "operations":
            return self.source.component_raw_operations(
                self.conductor_counter, self.sheet_name, self.identifier
            )
        return self.source.component_raw_inputs(
            self.conductor_counter, self.sheet_name, self.identifier
        )

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

        return finalize_solid_operations(StrandComponentOperations(
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
            # Strain rows may be absent for a pure stabilizer strand (strain
            # is only read on superconductor paths, which skip stabilizers).
            strain_mode=int(wb.get("IEPS", 0)),
            strain_value=float(wb.get("EPS", 0.0)),
            tcs_evaluation=bool(wb.get("TCS_EVALUATION", False)),
            fix_potential_flag=bool(wb["FIX_POTENTIAL_FLAG"]),
            fix_potential_number=int(wb["FIX_POTENTIAL_NUMBER"]),
            fix_potential_coordinate=wb["FIX_POTENTIAL_COORDINATE"],
            fix_potential_value=wb["FIX_POTENTIAL_VALUE"],
            # Optional AC-loss / proportional-field keys (same handling
            # as StrandMixedComponent; silently defaulting them dropped
            # the deck values for stabilizers - seen 2026-09-05).
            magnetic_field_scales_with_current=bool(
                wb.get("B_SCALES_WITH_CURRENT", False)
            ),
            coupling_loss_time_constant=loss_constant(
                wb.get("COUPLING_LOSS_TIME_CONSTANT", 0.0)
            ),
            coupling_loss_relaxation_time=loss_constant(
                wb.get("COUPLING_LOSS_RELAXATION_TIME", 0.0)
            ),
            coupling_loss_copper_scaling=bool(
                wb.get("COUPLING_LOSS_COPPER_SCALING", False) or False
            ),
            coupling_loss_reference_temperature=float(
                wb.get("COUPLING_LOSS_REFERENCE_TEMPERATURE", 0.0) or 0.0
            ),
            coupling_loss_reference_field=float(
                wb.get("COUPLING_LOSS_REFERENCE_FIELD", 0.0) or 0.0
            ),
            tape_hysteresis_loss=bool(
                wb.get("TAPE_HYSTERESIS_LOSS", False) or False
            ),
            eddy_loss_geometry_constant=float(
                wb.get("EDDY_LOSS_GEOMETRY_CONSTANT", 0.0) or 0.0
            ),
            filament_diameter=float(
                wb.get("FILAMENT_DIAMETER", 0.0) or 0.0
            ),
            transverse_coupling_file=str(
                wb.get("TRANSVERSE_COUPLING_FILE", "") or ""
            ),
        ), self.identifier)
