"""
Input dataclasses and loader for StrandMixedComponent.

StrandMixedComponentInputs – all input parameters for a StrandMixedComponent
StrandMixedInputLoader      – reads Excel and constructs the dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass



from opensc2.components.solid.solid_component_inputs import (
    SolidComponentInputs,
    StrandComponentOperations,
    loss_constant,
)
from opensc2.conductor.conductor_flags import InterpolationType
from opensc2.thermal.thermal_flags import HeatExcitation
from opensc2.electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode


@dataclass
class StrandMixedComponentInputs(SolidComponentInputs):
    """Input parameters for StrandMixedComponent (mixed SC + stabilizer strand)."""

    # Superconducting material parameters (used in strand_component.py)
    superconducting_material: str             # superconducting_material (lowercased by loader)
    upper_critical_field_at_0K: float         # Bc20m — Bc2 at T=0, B=0 in T
    critical_current_scaling_constant: float  # c0    — mutated at runtime (INGEGNERISTIC_MODE)
    critical_temperature_at_0T: float         # Tc0m  — Tc0 at B=0 in K

    # Stabilizer material
    stabilizer_material: str                  # stabilizer_material (lowercased by loader)

    # Strand geometry
    num_sc_strands: int                       # N_sc_strand
    sc_strand_diameter: float                 # d_sc_strand in m
    num_stabilizer_strands: int               # N_stab_strand
    stabilizer_strand_diameter: float         # d_stab_strand in m

    # Stabilizer-to-SC ratio definition
    stabilizer_to_sc_ratio_mode: int          # ISTAB_NON_STAB — how STAB_NON_STAB is defined
    stabilizer_to_sc_ratio: float             # STAB_NON_STAB

    # Critical current parameters
    critical_current_definition_mode: int     # C0_MODE — physical or ingegneristic definition

    # Homogenization / material composition
    num_material_types: int                   # NUM_MATERIAL_TYPES

    # Electrical model parameters
    flux_flow_electric_field: float           # E0 in V/m
    power_law_exponent: float                 # nn (n-value exponent)
    residual_resistivity_ratio: float         # RRR (Optional: only meaningful for Cu stabilizer)
