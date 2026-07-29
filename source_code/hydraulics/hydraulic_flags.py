"""
This module contains all flags regarding hydraulic or fluid properties.
"""

from enum import Enum, IntEnum, auto


class FluidType(Enum):
    """Coolants; the value is the CoolProp fluid name."""
    HELIUM = "helium"
    NITROGEN = "nitrogen"


def get_fluid_type(flag: str) -> FluidType:
    try:
        return FluidType(flag.lower())
    except ValueError:
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


class HydraulicFormulation(Enum):
    """Primary variables of the 1D channel hydraulics.

    VELOCITY solves the classical (v, p, T) primitive-variable system;
    MASS_FLOW solves the similarity-transformed (mdot, p, T) system, whose
    first fluid unknown per channel is the mass flow rate mdot = rho*A*v
    (the native coupling variable of the hydraulic network). AUTO is the
    input default and resolves at setup time: network coupling declared ->
    MASS_FLOW, otherwise VELOCITY (see hydraulics/formulation.py).
    """
    AUTO = "auto"
    VELOCITY = "velocity"
    MASS_FLOW = "mass_flow"


def get_hydraulic_formulation(flag: str) -> HydraulicFormulation:
    try:
        return HydraulicFormulation(str(flag).lower())
    except ValueError:
        raise ValueError(
            f"Unknown hydraulic formulation {flag!r}: valid values are "
            "'auto', 'velocity' and 'mass_flow'."
        )
