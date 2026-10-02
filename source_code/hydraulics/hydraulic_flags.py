"""
This module contains all flags regarding hydraulic or fluid properties.
"""

from enum import Enum, IntEnum, auto


class FluidType(Enum):
    """Coolants; the value is the CoolProp fluid name.

    CONSTANT is not a CoolProp fluid: it selects the constant-property fluid
    registered in interfaces.coolprop_interface (see ConstantFluidProperties),
    used by the verification suite to compare against closed-form solutions.
    """
    HELIUM = "helium"
    NITROGEN = "nitrogen"
    CONSTANT = "constant"


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

    MASS_FLOW_EXPLICIT is the same (mdot, p, T) unknown set, assembled
    directly from the written-out coefficients (hydraulics/
    mass_flow_equations.py) instead of the runtime transform. It is not
    an input value of ``hydraulic_formulation``: it is selected by the
    boolean input ``explicit_mass_flow_formulation`` on top of a
    mass-flow formulation (resolved in hydraulics/formulation.py).
    """
    AUTO = "auto"
    VELOCITY = "velocity"
    MASS_FLOW = "mass_flow"
    MASS_FLOW_EXPLICIT = "mass_flow_explicit"


# Formulations whose first fluid unknown per channel is the mass flow rate
# (both share the mass-flow boundary conditions, initial-condition seeding,
# network port coupling and solution reorganization).
MASS_FLOW_FORMULATIONS = (
    HydraulicFormulation.MASS_FLOW,
    HydraulicFormulation.MASS_FLOW_EXPLICIT,
)


def get_hydraulic_formulation(flag: str) -> HydraulicFormulation:
    try:
        formulation = HydraulicFormulation(str(flag).lower())
    except ValueError:
        formulation = None
    if formulation is None or formulation is HydraulicFormulation.MASS_FLOW_EXPLICIT:
        raise ValueError(
            f"Unknown hydraulic formulation {flag!r}: valid values are "
            "'auto', 'velocity' and 'mass_flow' (the explicit mass-flow "
            "assembly is selected with explicit_mass_flow_formulation: true)."
        )
    return formulation
