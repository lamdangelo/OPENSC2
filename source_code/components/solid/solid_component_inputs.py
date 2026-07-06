"""
Shared input dataclasses and operations dataclasses for all solid components.

SolidComponentInputs    – base inputs (cross_section, cos_theta)
SolidComponentOperations – operations fields common to all solid components
StrandComponentOperations – operations fields additional to strand-type components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from conductor.conductor_flags import InterpolationType
from electromagnetics.electromagnetic_flags import (
    CurrentMode, 
    BFieldDefinitionType
)
from thermal.thermal_flags import HeatExcitation


@dataclass
class SolidComponentInputs:
    """Input parameters shared by all solid components (accessed in solid_component.py)."""
    cross_section: float   # CROSSECTION — total perpendicular cross section in m²
    cos_theta: float       # COSTETA     — cos(θ) of inclination wrt jacket axis
    x_barycenter: float    # X_barycenter — x coordinate of the barycenter in m
    y_barycenter: float    # Y_barycenter — y coordinate of the barycenter in m
    show_figure: bool      # Show_fig    — whether to show the component's real-time plots


@dataclass
class SolidComponentOperations:
    """Operations parameters common to all solid components (accessed in solid_component.py)."""

    # Magnetic field boundary condition
    magnetic_field_bc_mode: BFieldDefinitionType  # IBIFUN   — mode flag for B-field BC
    magnetic_field_units: Optional[str]   # B_field_units — "T/A" or None; set to None after processing
    magnetic_field_inlet_initial: float   # BISS     — B at conductor inlet, initial value
    magnetic_field_outlet_initial: float  # BOSS     — B at conductor outlet, initial value
    magnetic_field_inlet_transient: float # BITR     — B at conductor inlet, transient value
    magnetic_field_outlet_transient: float# BOTR     — B at conductor outlet, transient value
    magnetic_field_interpolation: InterpolationType  # B_INTERPOLATION

    # Operating current
    operating_current_mode: CurrentMode   # IOP_MODE — mutated to None by deal_with_flag_IOP_MODE
    operating_current_interpolation: InterpolationType  # IOP_INTERPOLATION

    # External heat source
    heat_flux_mode: HeatExcitation        # IQFUN    — heat source/flux mode flag
    heat_flux_time_start: float           # TQBEG    — time at which heat flux begins
    heat_flux_time_end: float             # TQEND    — time at which heat flux ends
    heat_flux_amplitude: float            # Q0       — amplitude of imposed heat flux
    heat_flux_position_start: float       # XQBEG    — start position of heat flux
    heat_flux_position_end: float         # XQEND    — end position of heat flux
    heat_flux_interpolation: InterpolationType  # Q_INTERPOLATION

    # Initial temperature spatial distribution
    initial_temperature_mode: int         # INTIAL — 0: weighted average of contacting channels; ±1: user defined
    inlet_temperature: float              # TEMINL — temperature at conductor inlet in K
    outlet_temperature: float             # TEMOUT — temperature at conductor outlet in K


@dataclass
class StrandComponentOperations(SolidComponentOperations):
    """Additional operations parameters for strand-type components (accessed in strand_component.py)."""

    # Magnetic field gradient (alpha_B)
    alpha_b_mode: int                     # IALPHAB            — mode flag for alpha_B coefficient
    alpha_b_interpolation: str            # ALPHAB_INTERPOLATION

    # Strain (epsilon)
    strain_mode: int                      # IEPS               — strain boundary condition mode
    strain_value: float                   # EPS                — constant strain (used when IEPS == 1)

    # Current sharing temperature evaluation
    tcs_evaluation: bool                  # TCS_EVALUATION — flag to trigger Tcs evaluation

    # Fixed electric potential
    fix_potential_flag: bool              # FIX_POTENTIAL_FLAG
    fix_potential_number: int             # FIX_POTENTIAL_NUMBER
    fix_potential_coordinate: Any         # FIX_POTENTIAL_COORDINATE — float/str from Excel,
    fix_potential_value: Any              # FIX_POTENTIAL_VALUE      — converted to ndarray or None
