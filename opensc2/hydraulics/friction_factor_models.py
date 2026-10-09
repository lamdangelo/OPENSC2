"""
This module contains functionalities regarding the hydraulic friction model.
"""

from dataclasses import dataclass 
from enum import IntEnum, auto
import numpy as np 


class FrictionFactorModelType(IntEnum):
    """
    Enumerates the different friction factor model types.
    """
    # For fluid components of kind "hole"
    ITER_CONDUCTOR_79 = auto()  
    ITER_CONDUCTOR_810 = auto()
    TURBULENT_FRICTION_WITH_NEWTON_V1 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V2 = auto()  
    TURBULENT_FRICTION_WITH_NEWTON_V3 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V4 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V5 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V6 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V7 = auto() 
    TURBULENT_FRICTION_WITH_NEWTON_V8 = auto() 
    BLASIUS = auto()  
    BHATTI_SHAH = auto() 
    LHC_MAGNET = auto()
    BESSETTE_ITER_CS_SPIRAL = auto()
    CSI_LS_BEST_FIT = auto()
    TRONZA_ITER_TF_HOLE = auto()
    INCROPERA_RECTANGULAR_TF = auto()
    INCROPERA_RECTANGULAR_CS = auto()
    FLAT_SPIRAL = auto()
    DUCT_DEMO_COMMON_RECTANGULAR = auto()
    DUCT_DEMO_COMMON_TRIANGULAR = auto()
    HAALAND = auto()
    COLEBROOK = auto()
    PETUKHOV = auto()
    ENEA_HTS_CICC = auto()

    # For fluid components of kind "bundle"
    TRONZA_ITER_TF_BUNDLE = auto()
    NICOLLET = auto()
    WANNER = auto()
    KSTAR = auto()
    KATHEDER_EAST = auto()
    KATHEDER_PURE = auto()
    DARCY_FORCHHEIMER_POROUS_MEDIUM = auto()
    LHC_PONCET = auto()
    ITER_CS = auto()
    DTT = auto()
    DEMO_TF_HTS = auto()
    HTS_CL = auto()

    # User-defined friction factor 
    USER_DEFINED = auto()


    @staticmethod 
    def get_friction_factor_model(flag: int):
        """Convert friction factor model flag from Excel input file to FrictionFactorModelType."""
        if flag == 100:
            return FrictionFactorModelType.ITER_CONDUCTOR_79 
        elif flag == 101:
            return FrictionFactorModelType.ITER_CONDUCTOR_810
        elif flag == 102:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V1 
        elif flag == 103:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V2
        elif flag == 104:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V3 
        elif flag == 105:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V4 
        elif flag == 106:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V5 
        elif flag == 107:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V6 
        elif flag == 108:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V7 
        elif flag == 109:
            return FrictionFactorModelType.TURBULENT_FRICTION_WITH_NEWTON_V8 
        elif flag == 110:
            return FrictionFactorModelType.BLASIUS 
        elif flag == 111:
            return FrictionFactorModelType.BHATTI_SHAH
        elif flag == 112:
            return FrictionFactorModelType.LHC_MAGNET
        elif flag == 113:
            return FrictionFactorModelType.BESSETTE_ITER_CS_SPIRAL
        elif flag == 114:
            return FrictionFactorModelType.CSI_LS_BEST_FIT
        elif flag == 115:
            return FrictionFactorModelType.TRONZA_ITER_TF_HOLE
        elif flag == 116:
            return FrictionFactorModelType.INCROPERA_RECTANGULAR_TF
        elif flag == 117:
            return FrictionFactorModelType.INCROPERA_RECTANGULAR_CS
        elif flag == 118:
            return FrictionFactorModelType.FLAT_SPIRAL
        elif flag == 119:
            return FrictionFactorModelType.DUCT_DEMO_COMMON_RECTANGULAR
        elif flag == 120:
            return FrictionFactorModelType.DUCT_DEMO_COMMON_TRIANGULAR
        elif flag == 121:
            return FrictionFactorModelType.HAALAND
        elif flag == 122:
            return FrictionFactorModelType.COLEBROOK
        elif flag == 123:
            return FrictionFactorModelType.PETUKHOV
        elif flag == 124:
            return FrictionFactorModelType.ENEA_HTS_CICC
        elif flag == 200:
            return FrictionFactorModelType.TRONZA_ITER_TF_BUNDLE
        elif flag == 201:
            return FrictionFactorModelType.NICOLLET
        elif flag == 202:
            return FrictionFactorModelType.WANNER
        elif flag == 203:
            return FrictionFactorModelType.KSTAR 
        elif flag == 204:
            return FrictionFactorModelType.KATHEDER_EAST
        elif flag == 205:
            return FrictionFactorModelType.KATHEDER_PURE
        elif flag == 206: 
            return FrictionFactorModelType.DARCY_FORCHHEIMER_POROUS_MEDIUM
        elif flag == 207:
            return FrictionFactorModelType.LHC_PONCET
        elif flag == 208:
            return FrictionFactorModelType.ITER_CS
        elif flag == 209:
            return FrictionFactorModelType.DTT 
        elif flag == 210:
            return FrictionFactorModelType.DEMO_TF_HTS
        elif flag == 211:
            return FrictionFactorModelType.HTS_CL
        elif flag == -99:
            return FrictionFactorModelType.USER_DEFINED
        else:
            ValueError(f"Unknown friction factor model flag {flag}.")


@dataclass
class FrictionFactors:
    """
    Stores the laminar, turbulent and total friction factors.
    """
    laminar: np.ndarray | float | None = None 
    turbulent: np.ndarray | float | None = None 
    total: np.ndarray | float = 0.0 


    @classmethod 
    def initialize(cls, reynolds: np.ndarray, guess: np.ndarray=None):
        """
        Initializes the laminar, turbulent and total friction factors based on the shape of the
        Reynolds number.
        """
        laminar_friction_factor = np.zeros_like(reynolds)
        turbulent_friction_factor = np.zeros_like(reynolds)
        if guess is None:
            total_friction_factor = np.zeros_like(reynolds)
        else:
            total_friction_factor = guess * np.ones_like(reynolds)
        return cls(
            laminar=laminar_friction_factor,
            turbulent=turbulent_friction_factor,
            total=total_friction_factor
        )
