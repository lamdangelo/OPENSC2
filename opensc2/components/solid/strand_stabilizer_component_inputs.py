"""
Input dataclasses and loader for StrandStabilizerComponent.

StrandStabilizerComponentInputs – all input parameters for a StrandStabilizerComponent
StrandStabilizerInputLoader      – reads Excel and constructs the dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional



from opensc2.components.solid.solid_component_inputs import (
    SolidComponentInputs,
    StrandComponentOperations,
    loss_constant,
)
from opensc2.conductor.conductor_flags import InterpolationType
from opensc2.thermal.thermal_flags import HeatExcitation
from opensc2.electromagnetics.electromagnetic_flags import BFieldDefinitionType, CurrentMode


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
