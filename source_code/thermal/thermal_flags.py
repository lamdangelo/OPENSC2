"""
This module contains all flags regarding thermal properties.
"""

from enum import IntEnum


class HeatExcitation(IntEnum):  # IQFUN
    """Values match the IQFUN flag of the Excel operation workbook."""
    NO_HEATING = 0
    SQUARE_WAVE_IN_TIME_AND_SPACE = 1
    FROM_FILE = -1
    FROM_USER_FUNCTION = -2


    @staticmethod
    def get_heat_excitation(flag: int):
        """
        Translates the Excel integer flag of IQFUN to a HeatExcitation object.
        """
        try:
            return HeatExcitation(flag)
        except ValueError:
            raise ValueError(f"Heat excitation (IQFUN) flag {flag} not known.")