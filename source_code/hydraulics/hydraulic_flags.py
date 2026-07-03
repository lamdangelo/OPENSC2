"""
This module contains all flags regarding hydraulic or fluid properties.
"""

from enum import Enum, IntEnum, auto


class FluidType(Enum):
    HELIUM = "helium"


def get_fluid_type(flag: str) -> FluidType:
    if flag.lower() == "helium":
        return FluidType.HELIUM
    else:
        raise ValueError(f"Unknown fluid type {flag}.")


class ChannelType(Enum):
    HOLE = "hole"
    BUNDLE = "bundle"


def get_channel_type(flag: str) -> ChannelType:
    if flag.lower() == "hole":
        return ChannelType.HOLE
    elif flag.lower() == "bundle":
        return ChannelType.BUNDLE
    else:
        raise ValueError(f"Unknown channel type {flag}.")


class FlowDirection(IntEnum):
    FORWARD = auto()
    BACKWARD = auto()


def get_flow_direction(flag: str) -> FlowDirection:
    if flag.lower() == "forward":
        return FlowDirection.FORWARD
    elif flag.lower() == "backward":
        return FlowDirection.BACKWARD
    else:
        raise ValueError(f"Unknown flow direction {flag}.")


class HydraulicBC(IntEnum):
    IMPOSE_PRESSURE_DROP = auto()
    IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY = auto()
    IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE = auto()


def get_hydraulic_bc(flag: int) -> HydraulicBC:
    """Convert the INTIAL flag from the Excel file to a HydraulicBC.

    The sign of INTIAL states whether the boundary condition values come from
    the operations workbook (positive) or from the external flow file
    (negative); it is stored separately as
    ``FluidComponentOperations.bc_values_from_file``. Only the magnitude
    selects the kind of boundary condition.
    """
    if abs(flag) == 1:
        return HydraulicBC.IMPOSE_PRESSURE_DROP
    elif abs(flag) == 2:
        return HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY
    elif abs(flag) == 3:
        return HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE
    else:
        raise ValueError(f"Hydraulic BC flag {flag} not known.")
