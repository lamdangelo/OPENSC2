"""
This module contains all flags regarding electromagnetic properties.
"""

from enum import Enum, IntEnum, auto


class BFieldDefinitionType(IntEnum):
    """Values match the IBIFUN flag of the Excel operation workbook."""
    FROM_FILE = -1
    CONSTANT_OR_LINEAR = 0
    LINEAR_WITH_TRANSIENT = 1

    @staticmethod
    def get_bfield_definition_type(flag: int):
        """
        Translates the Excel input value IBIFUN to a BFieldDefinitionType object.
        """
        try:
            return BFieldDefinitionType(flag)
        except ValueError:
            raise ValueError(f"B-field definition type (IBIFUN) {flag} is unknown.")


class CurrentMode(IntEnum):
    """
    An enumeration class to define flags for the current mode, in the Excel file
    known as I0_OP_MODE.
    """
    CURRENT_NOT_DEFINED = auto()
    CURRENT_IS_CONSTANT = auto()
    CURRENT_IS_FROM_FILE = auto()
    CURRENT_IS_FUNCTION = auto()

    @staticmethod
    def get_current_mode_flag(iop_value: int):
        """Convert the I0_OP_MODE value from the Excel file to a CurrentMode."""
        if iop_value is None or (
            isinstance(iop_value, str) and iop_value.lower() == "none"
        ):
            return CurrentMode.CURRENT_NOT_DEFINED
        elif iop_value == 0:
            return CurrentMode.CURRENT_IS_CONSTANT
        elif iop_value == -1:
            return CurrentMode.CURRENT_IS_FROM_FILE
        elif iop_value == -2:
            return CurrentMode.CURRENT_IS_FUNCTION
        else:
            raise ValueError(f"Invalid I0_OP_MODE value: {iop_value}")


class InductanceMode(IntEnum):
    """
    An enumeration class to define flags for the inductance mode, in the Excel file
    known as INDUCTANCE_MODE.
    """
    ANALYTICAL = auto()
    APPROXIMATED = auto()

    @staticmethod
    def get_inductance_mode_flag(inductance_mode: int):
        """Convert the INDUCTANCE_MODE value from the Excel file to an InductanceMode."""
        if inductance_mode == 0:
            return InductanceMode.ANALYTICAL
        elif inductance_mode == 1:
            return InductanceMode.APPROXIMATED
        else:
            raise ValueError(f"Invalid INDUCTANCE_MODE value: {inductance_mode}")


class SelfInductanceMode(IntEnum):
    """
    An enumeration class to define flags for the self-inductance mode, in the Excel file
    known as SELF_INDUCTANCE_MODE.
    """
    MODE_1 = auto()
    MODE_2 = auto()

    @staticmethod
    def get_self_inductance_mode_flag(self_inductance_mode: int):
        """Convert the SELF_INDUCTANCE_MODE value from the Excel file to a SelfInductanceMode."""
        if self_inductance_mode == 1:
            return SelfInductanceMode.MODE_1
        elif self_inductance_mode == 2:
            return SelfInductanceMode.MODE_2
        else:
            raise ValueError(f"Invalid SELF_INDUCTANCE_MODE value: {self_inductance_mode}")


class ElectricSolver(IntEnum):
    """
    An enumeration class to define flags for the electric solver, in the Excel file
    known as ELECTRIC_SOLVER.
    """
    STEADY_STATE = auto()
    TRANSIENT = auto()

    @staticmethod
    def get_electric_solver_flag(electric_solver: int):
        """Convert the ELECTRIC_SOLVER value from the Excel file to an ElectricSolver."""
        if electric_solver == 0:
            return ElectricSolver.STEADY_STATE
        elif electric_solver == 1:
            return ElectricSolver.TRANSIENT
        else:
            raise ValueError(f"Invalid ELECTRIC_SOLVER value: {electric_solver}")


class ElectricConductanceMode(IntEnum):
    """
    An enumeration class to define the kind of electric conductance, in the Excel file
    known as electric_conductance_mode.
    """
    NO_CONDUCTANCE = auto()
    CONDUCTANCE_PER_UNIT_LENGTH = auto()
    ACTUAL_CONDUCTANCE_BETWEEN_COMPONENTS = auto()

    @staticmethod
    def get_electric_conductance_mode(flag: int):
        """
        Convert the electric_conductance_mode value from the Excel file to an
        ElectricConductanceMode.
        """
        if flag == 0:
            return ElectricConductanceMode.NO_CONDUCTANCE
        elif flag == 1:
            return ElectricConductanceMode.CONDUCTANCE_PER_UNIT_LENGTH
        elif flag == 2:
            return ElectricConductanceMode.ACTUAL_CONDUCTANCE_BETWEEN_COMPONENTS
        else:
            raise ValueError(f"Invalid electric_conductance_mode value: {flag}")