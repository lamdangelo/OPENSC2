"""
This module contains heat transfer coefficient models.
"""

from enum import IntEnum, auto


class HeatTransferModelType(IntEnum):
    """
    Enumerates the different heat transfer coefficient types.
    """
    DITTUS_BOELTER_LOWER_LIMIT_1 = auto()
    DITTUS_BOELTER_LOWER_LIMIT_119 = auto()
    DITTUS_BOELTER_LOWER_LIMIT_120 = auto()
    DITTUS_BOELTER_LOWER_LIMIT_121 = auto()  # TODO: find more reasonable names
    CORRELATION_211 = auto()
    RECTANGULAR_DUCT_ENEA_HTS_CICC = auto()
    # Bare Dittus-Boelter with the heating exponent Pr^0.4 and NO lower
    # bound on the Nusselt number (h -> 0 in stagnant flow), matching the
    # THEA Dittus-Boelter model.
    DITTUS_BOELTER_PURE = auto()


    @staticmethod
    def get_heat_transfer_model(flag: int):
        """Translates the Excel file flag into a HeatTransferModel."""
        if flag == 1:
            return HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_1
        elif flag == 2:
            return HeatTransferModelType.DITTUS_BOELTER_PURE
        elif flag == 119:
            return HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_119
        elif flag == 120:
            return HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_120
        elif flag == 121:
            return HeatTransferModelType.DITTUS_BOELTER_LOWER_LIMIT_121
        elif flag == 211:
            return HeatTransferModelType.CORRELATION_211
        elif flag == 212:
            return HeatTransferModelType.RECTANGULAR_DUCT_ENEA_HTS_CICC
        else:
            raise ValueError(f"Unknwn heat transfer model {flag}.")