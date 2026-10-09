"""
Input dataclasses and loader for StackComponent.

TapeLayer               – one material layer in the HTS tape stack
StackComponentInputs    – all input parameters for a StackComponent
StackComponentInputLoader – reads Excel and constructs StackComponentInputs + StrandComponentOperations
"""

from __future__ import annotations

from dataclasses import dataclass, field



from opensc2.components.solid.solid_component_inputs import (
    SolidComponentInputs,
    StrandComponentOperations,
    loss_constant,
)
from opensc2.conductor.conductor_flags import InterpolationType
from opensc2.thermal.thermal_flags import HeatExcitation
from opensc2.electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode


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
