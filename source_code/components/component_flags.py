"""
This module contains the component types as string enumerations equal to the Excel naming 
convention.
"""

from enum import Enum


class ComponentType(Enum):
    FLUID = "CHAN"
    STACK = "STACK"
    STRAND_MIXED = "STR_MIX"
    STRAND_STABILIZER = "STR_STAB"
    JACKET = "Z_JACKET"


def get_component_type(component_name: str) -> ComponentType:
    if component_name == "CHAN":
        return ComponentType.FLUID 
    elif component_name == "STACK":
        return ComponentType.STACK 
    elif component_name == "STR_MIX":
        return ComponentType.STRAND_MIXED
    elif component_name == "STR_STAB":
        return ComponentType.STRAND_STABILIZER
    elif component_name == "Z_JACKET":
        return ComponentType.JACKET
    else:
        ValueError(f"Unknown component type {component_name}.")

