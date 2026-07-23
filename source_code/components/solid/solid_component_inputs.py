"""
Shared input dataclasses and operations dataclasses for all solid components.

SolidComponentInputs    – base inputs (cross_section, cos_theta)
SolidComponentOperations – operations fields common to all solid components
StrandComponentOperations – operations fields additional to strand-type components
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    # B_SCALES_WITH_CURRENT — a FROM_FILE spatial profile scales with
    # I(t)/I0 (proportional field model for spatially resolved profiles;
    # the linear counterpart is LINEAR_WITH_TRANSIENT).
    magnetic_field_scales_with_current: bool = field(default=False, kw_only=True)

    # TRANSVERSE_COUPLING_FILE — CSV of nonlocal transverse-conduction
    # patches (winding-geometry-adjacent positions exchanging heat
    # through the insulation, invisible to the 1D metric); "" disables.
    # See SolidComponent.get_transverse_coupling for the row format.
    transverse_coupling_file: str = field(default="", kw_only=True)

    # EDDY_LOSS_GEOMETRY_CONSTANT — geometry constant C [m^4] of the
    # eddy-current heat source p = sigma(T[,B]) * (dB/dt)^2 * C, where C
    # is the second moment of the conducting cross-section about its
    # centroid (for a long conductor in a transverse changing field; it
    # reduces to sigma * d^2 / 12 * A in the thin-wall limit). Applied to
    # the metallic jacket (sigma from jacket_material) and, on strands, to
    # the copper matrix. 0 disables the source. See
    # SolidComponent.get_eddy_loss.
    eddy_loss_geometry_constant: float = field(default=0.0, kw_only=True)


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

    # COUPLING_LOSS_TIME_CONSTANT — effective coupling time constant
    # n·tau in s for the AC coupling-loss heat source
    # p = (n·tau/µ0)·(dB/dt)² per unit strand volume; 0 disables the source.
    coupling_loss_time_constant: float = field(default=0.0, kw_only=True)

    # FILAMENT_DIAMETER — superconductor filament diameter d_f [m] for the
    # hysteresis (persistent-current magnetization) loss
    # p = (2/3pi) * Jc(B,T) * d_f * |dB/dt| per unit superconductor volume;
    # 0 disables the source. See StrandComponent.get_hysteresis_loss.
    filament_diameter: float = field(default=0.0, kw_only=True)
