"""
Input dataclasses and loader for JacketComponent.

JacketComponentInputs – all input parameters for a JacketComponent
JacketComponentInputLoader – reads Excel and constructs the dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass



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
