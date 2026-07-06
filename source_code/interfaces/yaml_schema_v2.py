"""
Schema v2 of the OPENSC2 YAML input format: descriptive leaf-key names.

One alias table per document section maps the v2 key onto the legacy
workbook variable name (see refactoring/yaml_schema_v2_key_names.md for the
approved naming rationale). The registry translates v2 keys back to the
legacy names it serves to the loaders, and UNKNOWN keys pass through
unchanged — so schema-v1 files (legacy keys) remain fully readable.

Special case: the multiplexed legacy ``INTIAL`` is split in v2 — fluid
components carry ``hydraulic_boundary_condition`` (magnitude) plus
``boundary_values_from_file`` (the legacy sign); solid components carry
``initial_temperature_mode``. :func:`recompose_legacy_intial` rebuilds the
signed integer.
"""

ENVIRONMENT_KEYS = {
    "medium": "Medium",
    "temperature": "Temperature",
    "pressure": "Pressure",
}

CONDUCTOR_INPUT_KEYS = {
    "length": "ZLENGTH",
    "diameter": "Diameter",
    "is_rectangular": "Is_rectangular",
    "width": "Width",
    "height": "Height",
    "is_joint": "ISJOINT",
    "current_mode": "I0_OP_MODE",
    "initial_current": "I0_OP_TOT",
    "inlet_heated_zone_start": "XJBEG",
    "inlet_heated_zone_end": "XJBEIN",
    "outlet_heated_zone_start": "XJBEOUT",
    "outlet_heated_zone_end": "XJENOUT",
    "thermohydraulic_method": "METHOD",
    "electric_method": "ELECTRIC_METHOD",
    "electric_time_step": "ELECTRIC_TIME_STEP",
    "upwind": "UPWIND",
    "phi_radiative": "Phi_rad",
    "phi_convective": "Phi_conv",
}

CONDUCTOR_OPERATION_KEYS = {
    "electric_solver": "ELECTRIC_SOLVER",
    "do_equipotential_surfaces_exist": "EQUIPOTENTIAL_SURFACE_FLAG",
    "number_of_equipotential_surfaces": "EQUIPOTENTIAL_SURFACE_NUMBER",
    "equipotential_surface_coordinates": "EQUIPOTENTIAL_SURFACE_COORDINATE",
    "inductance_mode": "INDUCTANCE_MODE",
    "self_inductance_mode": "SELF_INDUCTANCE_MODE",
    "maximum_iteration_number": "MAXIMUM_ITERATION_NUMBER",
}

# Component inputs of every kind share one table (v2 names are unique).
COMPONENT_INPUT_KEYS = {
    "cross_section": "CROSSECTION",
    "hydraulic_diameter": "HYDIAMETER",
    "cos_theta": "COSTETA",
    "void_fraction": "VOID_FRACTION",
    "fluid_type": "FLUID_TYPE",
    "channel_type": "CHANNEL_TYPE",
    "friction_factor_model": "IFRICTION",
    "friction_multiplier": "FRICTION_MULTIPLIER",
    "heat_transfer_model": "Flag_htc_steady_corr",
    "roughness": "Roughness",
    "is_rectangular": "ISRECTANGULAR",
    "width": "SIDE1",
    "height": "SIDE2",
    "x_barycenter": "X_barycenter",
    "y_barycenter": "Y_barycenter",
    "show_figure": "Show_fig",
    "number_of_material_types": "NUM_MATERIAL_TYPES",
    "superconductor_strand_count": "N_sc_strand",
    "superconductor_strand_diameter": "d_sc_strand",
    "stabilizer_strand_count": "N_stab_strand",
    "stabilizer_strand_diameter": "d_stab_strand",
    "stabilizer_to_superconductor_ratio_mode": "ISTAB_NON_STAB",
    "stabilizer_to_superconductor_ratio": "STAB_NON_STAB",
    "residual_resistivity_ratio": "RRR",
    "critical_current_definition_mode": "C0_MODE",
    "critical_current_scaling_constant": "c0",
    "critical_temperature_at_zero_field": "Tc0m",
    "upper_critical_field_at_zero_temperature": "Bc20m",
    "power_law_exponent": "nn",
    "electric_field_criterion": "E0",
    "tape_count": "N_tape",
    "tape_identifier": "Tape_number",
    "number_of_material_layers": "Material_number",
    "stack_width": "Stack_width",
    "jacket_kind": "Jacket_kind",
    "emissivity": "Emissivity",
    "outer_perimeter": "Outer_perimeter",
    "inner_perimeter": "Inner_perimeter",
    # already descriptive, listed for completeness of the reference:
    # stabilizer_material, superconducting_material, jacket_material,
    # insulation_material, jacket_cross_section, insulation_cross_section
}

COMPONENT_OPERATION_KEYS = {
    "initial_temperature_mode": "INTIAL",  # solids; fluids use the split form
    "inlet_temperature": "TEMINL",
    "outlet_temperature": "TEMOUT",
    "initial_temperature": "TEMINI",
    "initial_temperature_outlet": "TEMINI_OUT",
    "inlet_pressure": "PREINL",
    "outlet_pressure": "PREOUT",
    "initial_pressure": "PREINI",
    "inlet_mass_flow_rate": "MDTIN",
    "outlet_mass_flow_rate": "MDTOUT",
    "flow_direction": "FLOWDIR",
    "heat_flux_mode": "IQFUN",
    "heat_flux_amplitude": "Q0",
    "heat_flux_time_start": "TQBEG",
    "heat_flux_time_end": "TQEND",
    "heat_flux_position_start": "XQBEG",
    "heat_flux_position_end": "XQEND",
    "heat_flux_interpolation": "Q_INTERPOLATION",
    "joule_heating_fraction": "QJFRACT",
    "magnetic_field_mode": "IBIFUN",
    "magnetic_field_inlet_initial": "BISS",
    "magnetic_field_outlet_initial": "BOSS",
    "magnetic_field_inlet_transient": "BITR",
    "magnetic_field_outlet_transient": "BOTR",
    "magnetic_field_interpolation": "B_INTERPOLATION",
    "magnetic_field_units": "B_field_units",
    "field_angle_mode": "IALPHAB",
    "field_angle_interpolation": "ALPHAB_INTERPOLATION",
    "fixed_field_angle_value": "fixAlphaBvalue",
    "strain_mode": "IEPS",
    "strain_value": "EPS",
    "strain_interpolation": "EPS_INTERPOLATION",
    "operating_current_mode": "IOP_MODE",
    "operating_current_interpolation": "IOP_INTERPOLATION",
    "current_sharing_temperature_evaluation": "TCS_EVALUATION",
    "fixed_potential_flag": "FIX_POTENTIAL_FLAG",
    "fixed_potential_number": "FIX_POTENTIAL_NUMBER",
    "fixed_potential_coordinates": "FIX_POTENTIAL_COORDINATE",
    "fixed_potential_values": "FIX_POTENTIAL_VALUE",
}

GRID_KEYS = {
    "number_of_elements": "NELEMS",
    "mesh_type": "ITYMSH",
    "refined_zone_number_of_elements": "NELREF",
    "refined_zone_start": "XBREFI",
    "refined_zone_end": "XEREFI",
    "minimum_element_size": "SIZMIN",
    "maximum_element_size": "SIZMAX",
    "growth_ratio_left": "DXINCRE_LEFT",
    "growth_ratio_right": "DXINCRE_RIGHT",
    "maximum_number_of_nodes": "MAXNOD",
}

COUPLING_PROPERTY_KEYS = {
    "contact_perimeter_mode": "contact_perimeter_flag",
    "heat_transfer_coefficient_mode": "HTC_choice",
    "contact_heat_transfer_coefficient": "contact_HTC",
    "heat_transfer_coefficient_multiplier": "HTC_multiplier",
    "open_perimeter_fraction": "open_perimeter_fract",
    "interface_thickness": "interf_thickness",
    "transverse_transport_multiplier": "trans_transp_multiplier",
}


def translate_to_legacy(mapping: dict, alias_table: dict) -> dict:
    """Rename v2 keys to their legacy names; unknown (v1/legacy or extra)
    keys pass through unchanged."""
    return {alias_table.get(key, key): value for key, value in mapping.items()}


def recompose_legacy_intial(operations: dict) -> dict:
    """Rebuild the multiplexed legacy INTIAL for fluid components from the
    v2 split keys (see module docstring); no-op if the split keys are
    absent (schema v1 or solid component)."""
    if "hydraulic_boundary_condition" not in operations:
        return operations
    operations = dict(operations)
    magnitude = abs(int(operations.pop("hydraulic_boundary_condition")))
    from_file = bool(operations.pop("boundary_values_from_file", False))
    operations["INTIAL"] = -magnitude if from_file else magnitude
    return operations


def invert(alias_table: dict) -> dict:
    """legacy -> v2 key mapping, for the converter."""
    return {legacy: v2 for v2, legacy in alias_table.items()}
