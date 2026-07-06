"""
This module defines flags for the conductor simulation.
"""

from enum import IntEnum, auto


class MethodFlag(IntEnum):
    """
    An enumeration class to define flags for the numerical method, in the Excel file
    known as METHOD.
    """
    BACKWARD_EULER = auto()
    CRANK_NICOLSON = auto()
    ADAMS_MOULTON_4TH_ORDER = auto()
    # Theta method with theta = 2/3 (as the Galerkin time-differencing method
    # of CryoSoft THEA): more accurate than backward Euler, less prone to
    # oscillations than Crank-Nicolson.
    GALERKIN = auto()
    # Variable-step backward differentiation formula of second order (BDF2),
    # A-stable and well suited to the stiff quench problem; the first time
    # step falls back to backward Euler to start the two-level history.
    BACKWARD_DIFFERENCE_2 = auto()


    @staticmethod
    def get_method_flag(method_value: str):
        """Convert the METHOD value from the Excel file to a MethodFlag."""
        if method_value == "BE":
            return MethodFlag.BACKWARD_EULER
        elif method_value == "CN":
            return MethodFlag.CRANK_NICOLSON
        elif method_value == "AM4":
            return MethodFlag.ADAMS_MOULTON_4TH_ORDER
        elif method_value == "GAL":
            return MethodFlag.GALERKIN
        elif method_value == "BDF2":
            return MethodFlag.BACKWARD_DIFFERENCE_2
        else:
            raise ValueError(f"Invalid METHOD value: {method_value}")


# One-step theta-family methods: the system matrix and known term are the
# usual theta blend between the current and the previous time level.
THETA_FAMILY_METHODS = (
    MethodFlag.BACKWARD_EULER,
    MethodFlag.CRANK_NICOLSON,
    MethodFlag.GALERKIN,
)

# All the methods that share the two-time-level array layout (load vector,
# external fluxes, radiative heat sources, solution history): the theta
# family plus BDF2, which is fully implicit in the loads but keeps two
# previous solution levels.
ONE_STEP_METHODS = THETA_FAMILY_METHODS + (MethodFlag.BACKWARD_DIFFERENCE_2,)


class ContactPerimeterFlag(IntEnum):
    """
    An enumeration class for the values of the contact_perimeter_flag sheet of
    the conductor coupling workbook: whether two components are in thermal
    contact and, if so, how their contact perimeter is defined.
    """
    NO_CONTACT = 0
    CONSTANT_CONTACT_PERIMETER = 1  # from sheet contact_perimeter of the coupling workbook
    VARIABLE_CONTACT_PERIMETER = -1  # interpolated from the external contact perimeter file


class InterpolationType(IntEnum):
    LINEAR = auto()
    CUBIC = auto()


    @staticmethod 
    def get_interpolation_type(flag: str):
        if flag == "linear":
            return InterpolationType.LINEAR
        elif flag == "cubic":
            return InterpolationType.CUBIC
        else:
            raise ValueError(f"Interpolation type {flag} is unknown.")


class ExternalFreeConvectionCorrelation(IntEnum):
    """
    An enumeration class to define flags for the external free convection correlation,
    in the Excel file known as external_free_convection_correlation.
    """
    VERTICAL_PLATE = auto()
    VERTICAL_PLATE_CHURCHILL_CHU = auto()
    VERTICAL_PLATE_CHURCHILL_CHU_ACCURATE = auto()
    LONG_HORIZONTAL_CYLINDER_MORGAN = auto()
    LONG_HORIZONTAL_CYLINDER_CHURCHILL_CHU = auto()


    @staticmethod
    def get_external_free_convection_correlation_flag(correlation_value: str):
        """
        Convert the external_free_convection_correlation value from the Excel file to an
        ExternalFreeConvectionCorrelation.
        """
        if correlation_value == "vertical_plate":
            return ExternalFreeConvectionCorrelation.VERTICAL_PLATE
        elif correlation_value == "vertical_plate_churchill_chu":
            return ExternalFreeConvectionCorrelation.VERTICAL_PLATE_CHURCHILL_CHU
        elif correlation_value == "vertical_plate_churchill_chu_accurate":
            return ExternalFreeConvectionCorrelation.VERTICAL_PLATE_CHURCHILL_CHU_ACCURATE
        elif correlation_value == "long_horizontal_cylinder_morgan":
            return ExternalFreeConvectionCorrelation.LONG_HORIZONTAL_CYLINDER_MORGAN
        elif correlation_value == "long_horizontal_cylinder_churchill_chu":
            return ExternalFreeConvectionCorrelation.LONG_HORIZONTAL_CYLINDER_CHURCHILL_CHU
        else:
            raise ValueError(f"Invalid external_free_convection_correlation value: {correlation_value}")


class HTC_Choice(IntEnum):
    """
    An enumeration class to define flags for the choice of heat transfer coefficient,
    in the Excel file known as HTC_choice.
    """
    NO_HTC = auto()
    CONDUCTION_HTC_COMPUTED = auto()
    CONDUCTION_HTC_READ_FROM_FILE = auto()
    CONVECTION_HTC_COMPUTED = auto()
    CONVECTION_HTC_READ_FROM_FILE = auto()
    RADIATIVE_HTC_COMPUTED = auto()
    RADIATIVE_HTC_READ_FROM_FILE = auto()
    MIXED_CONVECTION_AND_RADIATION_COMPUTED = auto()
    MIXED_CONVECTION_AND_RADIATION_FROM_FILE = auto()


    @staticmethod
    def get_htc_choice_flag(htc_choice: int):
        """Convert the HTC_choice value from the Excel file to a HTC_Choice."""
        if htc_choice == 0:
            return HTC_Choice.NO_HTC
        elif htc_choice == 1:
            return HTC_Choice.CONDUCTION_HTC_COMPUTED
        elif htc_choice == -1:
            return HTC_Choice.CONDUCTION_HTC_READ_FROM_FILE
        elif htc_choice == 2:
            return HTC_Choice.CONVECTION_HTC_COMPUTED
        elif htc_choice == -2:
            return HTC_Choice.CONVECTION_HTC_READ_FROM_FILE
        elif htc_choice == 3:
            return HTC_Choice.RADIATIVE_HTC_COMPUTED
        elif htc_choice == -3:
            return HTC_Choice.RADIATIVE_HTC_READ_FROM_FILE
        elif htc_choice == 4:
            return HTC_Choice.MIXED_CONVECTION_AND_RADIATION_COMPUTED
        elif htc_choice == -4:
            return HTC_Choice.MIXED_CONVECTION_AND_RADIATION_FROM_FILE
        else:
            raise ValueError(f"Invalid HTC_Choice value: {htc_choice}")
