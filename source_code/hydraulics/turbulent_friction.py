# turbulent.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import numpy as np

from hydraulics.auxiliary_functions import (
    evaluate_correction_diameter_hole
)


# ---------------------------------------------------------------------------
# Base API
# ---------------------------------------------------------------------------

class TurbulentCorrelation(ABC):
    """Base class for turbulent friction-factor correlations."""

    @abstractmethod
    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        pass


# ---------------------------------------------------------------------------
# Generic correlations
# ---------------------------------------------------------------------------

class Blasius(TurbulentCorrelation):
    """
    Darcy:
        f = 0.316 * Re^-0.25

    Converted to Fanning:
        f / 4
    """

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.316 * reynolds ** (-0.25) / 4.0
    

class BhattiShah(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.00128 + 0.1143 * np.maximum(reynolds, 4000) ** (-0.311)


@dataclass(slots=True)
class Haaland(TurbulentCorrelation):
    roughness: float
    hydraulic_diameter: float

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:

        relative_roughness = (
            self.roughness
            / self.hydraulic_diameter
        )

        return (
            (
                -1.8
                * np.log10(
                    (relative_roughness / 3.7) ** 1.11
                    + 6.9 / reynolds
                )
            )
            ** -2
        ) / 4.0


@dataclass(slots=True)
class Colebrook(TurbulentCorrelation):
    roughness: float
    hydraulic_diameter: float
    tolerance: float = 1e-6
    max_iterations: int = 100

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:

        friction = Haaland(
            self.roughness,
            self.hydraulic_diameter,
        )(reynolds)

        for _ in range(self.max_iterations):

            previous = friction.copy()

            friction = (
                (
                    -2.0
                    * np.log10(
                        (
                            self.roughness
                            / self.hydraulic_diameter
                            / 3.7
                        )
                        + (
                            2.51
                            / reynolds
                            / np.sqrt(previous)
                        )
                    )
                )
                ** -2
            ) / 4.0

            if np.max(np.abs(friction - previous)) < self.tolerance:
                return friction

        warnings.warn(
            "Colebrook solver failed to converge.",
            RuntimeWarning,
        )

        return friction


class Petukhov(TurbulentCorrelation):

    RE_MIN = 3e3
    RE_MAX = 5e6

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:

        valid = (
            (reynolds >= self.RE_MIN)
            & (reynolds <= self.RE_MAX)
        )

        if not np.all(valid):
            warnings.warn(
                "Petukhov correlation outside validity range.",
                RuntimeWarning,
            )

        return (
            (
                0.790 * np.log10(reynolds)
                - 1.64
            )
            ** -2
        ) / 4.0
    

class LHCMagnetRecipe(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:

        re = reynolds

        return np.select(
            [
                re <= 2e3,
                (re > 2e3) & (re < 4e3),
                (re >= 4e3) & (re < 3e4),
                re >= 3e4,
            ],
            [
                64.0 / re / 4.0,
                2.6471e-3 * re**0.3279 / 4.0,
                0.3194 * re**-0.25 / 4.0,
                1.0 / (1.8 * np.log10(re) - 1.64) ** 2 / 4.0,
            ],
        )
    

@dataclass(slots=True)
class BessetteIterCS2015(TurbulentCorrelation):

    hydraulic_diameter: float 

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        corrected_diameter = evaluate_correction_diameter_hole(self.hydraulic_diameter)
        corrected_reynolds = reynolds / corrected_diameter
        return np.where(
            corrected_reynolds < 1.5e5, 
            3.160 * corrected_reynolds ** (-0.249),
            0.216 * corrected_reynolds ** (-0.024)
        )
    

class CSILSBestFit(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.0958 * reynolds ** (-0.181)
    

@dataclass(slots=True)
class TronzaIterTFHole(TurbulentCorrelation):

    hydraulic_diameter: float 

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        # Different correction than in evaluate_correction_diameter_hole 
        corrected_diameter = 3.18e-3 / self.hydraulic_diameter 
        return (
            0.25 * 0.3164 * (reynolds / corrected_diameter) 
            ** (-0.25) / corrected_diameter
        )


@dataclass(slots=True)
class Wanner(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 6.1 * reynolds**(-0.51) / 4.0


@dataclass(slots=True)
class KSTAR(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.4335 * reynolds**(-0.263)


@dataclass(slots=True)
class Nicollet(TurbulentCorrelation):

    void_fraction: float
    perimeter_correction: float = 6.0 / 5.0

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:

        re = reynolds / self.perimeter_correction

        return (
            (
                0.0231
                + 19.5 / re**0.7953
            )
            / self.void_fraction**0.742
            / 4.0
        )
    

@dataclass(slots=True)
class KathederEast(TurbulentCorrelation):

    void_fraction: float 
    coef_a: float = 0.843
    coef_b: float = 0.0265

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return (
            self.void_fraction ** 0.72
            * (
                19.5 / reynolds ** self.coef_a
                + self.coef_b
            )
            / 4.0
        )
    

@dataclass(slots=True)
class KathederPure(TurbulentCorrelation):

    void_fraction: float 
    coef_a: float = 0.88
    coef_b: float = 0.051

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return (
            self.void_fraction ** 0.72
            * (
                19.5 / reynolds ** self.coef_a
                + self.coef_b
            )
            / 4.0
        )
    

class DarcyForchheimerPorousMedium(TurbulentCorrelation):

    def __init__(self, void_fraction: float, hydraulic_diameter: float):
        self.void_fraction = void_fraction 
        self.hydraulic_diameter = hydraulic_diameter
        self.coef_j = 2.42 / void_fraction ** 5.8
        self.coef_k = 19.6e-9 * void_fraction ** 3 / (1 - void_fraction) ** 2


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return (
            self.hydraulic_diameter ** 2 * self.void_fraction 
            / (2 * self.coef_k * reynolds)
            + self.hydraulic_diameter * self.void_fraction ** 2 / 2 
            * self.coef_j
        )
    

class DttBundle(TurbulentCorrelation):

    def __init__(self, void_fraction: float, hydraulic_diameter: float):
        self.void_fraction = void_fraction 
        self.hydraulic_diameter = hydraulic_diameter
        self.coef_j = 19.1 / void_fraction ** 4.23
        self.coef_k = 20.9e-9 * void_fraction ** 3 / (1 - void_fraction) ** 2  


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return (
            self.hydraulic_diameter ** 2 * self.void_fraction 
            / (2 * self.coef_k * reynolds )
            + self.hydraulic_diameter * self.void_fraction ** 2 / 2 
            * self.coef_j 
            * (self.hydraulic_diameter / self.void_fraction / np.sqrt(self.coef_k)) ** 0.14 
            / reynolds ** 0.14
        )      
    

class NewtonHole(TurbulentCorrelation):
    
    def __init__(self, number: int, hydraulic_diameter: float):
        self.coeff = NewtonCorrelationCoefficients(number)
        self.hydraulic_diameter = hydraulic_diameter
        self.pitch_to_height_ratio = self.coeff.gg / self.coeff.tthick1
        self.relative_roughness_height = (
            2.0 * self.coeff.tthick1 / hydraulic_diameter
            )
        

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        roughness_reynolds = self.coeff.tthick1 * self.hydraulic_diameter * reynolds 
        return evaluate_newton(
            hp=roughness_reynolds, 
            coeff=self.coeff,
            hd2=self.relative_roughness_height,
            goverh=self.pitch_to_height_ratio
        )
    

"""
ITER conductor correlations
"""
class IterConductor79(TurbulentCorrelation):
    """
    7/9 ITER spiral (average 6/8 - 8/10), from DDD 11, Magnet: Section 1. 
    Engineering Description, December 2004, pp. 32.
    """
    def __init__(self, hydraulic_diameter: float):
        self.hydraulic_diameter = hydraulic_diameter
        self.aa = 0.45
        self.bb = -0.034
        self.cc = 4.0
        self.dd = 5.0


    def __call__(self, reynolds: np.ndarray, tthick=1e-3) -> np.ndarray:
        _, corrected_diameter = self.evaluate_correction_diameter_hole(self.hydraulic_diameter, tthick)
        return (
            self.aa
            * (reynolds / corrected_diameter) ** self.bb
            / self.cc
            / corrected_diameter**self.dd
        )


class IterConductor810(TurbulentCorrelation):
    """
    8/10 ITER spiral, from N. Peng,L.Q. Liu, L. Serio, L.Y. Xiong, L. Zhang: 
    "Thermo-hydraulic analysis of the gradual cool-down to 80 K of the ITER toroidal field coil", 
    Cryogenics (49), 2009, pp.402-406.    
    """
    def __init__(self, hydraulic_diameter: float):
        self.hydraulic_diameter = hydraulic_diameter
        self.aa = 0.36
        self.bb = -0.038
        self.cc = 4.0
        self.dd = 5.0


    def __call__(self, reynolds: np.ndarray, tthick=1e-3) -> np.ndarray:
        _, corrected_diameter = self.evaluate_correction_diameter_hole(self.hydraulic_diameter, tthick)
        return (
            self.aa
            * (reynolds / corrected_diameter) ** self.bb
            / self.cc
            / corrected_diameter**self.dd
        )
    

class IncroperaRectangularTurbulentTFHole(TurbulentCorrelation):

    def __init__(self):
        self.thickness = 2.17e-4 
        self.width = 3.28e-3 
        thickness_width_ratio = self.thickness / self.width
        self.correction_factor = 2/3 + 11/24 * thickness_width_ratio * (2 - thickness_width_ratio)


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        corrected_reynolds = self.correction_factor * reynolds 
        return ( 0.79 * np.log10(corrected_reynolds) - 1.64 ) **(-2)
    

class IncroperaRectangularTurbulentCSHole(TurbulentCorrelation):

    def __init__(self, is_rectangular: bool, width: float=None, height: float=None, 
                 hydraulic_diameter: float=None, tthick: float=5e-4):
        if is_rectangular:
            self.side1 = width 
            self.side2 = height
        else:
            self.side1 = tthick
            self.side2 = evaluate_correction_diameter_hole(hydraulic_diameter, tthick)
        beta = min(self.side1, self.side2) / max(self.side1, self.side2)
        self.correction_factor = 2/3 + 11/24 * beta * (2 - beta)


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        corrected_reynolds = self.correction_factor * reynolds 
        return ( 0.79 * np.log10(corrected_reynolds) - 1.64 ) **(-2)
    

class FlatSpiral(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.1687 / 4.0 * reynolds ** (-0.1129)
    

class DuctDemoCommon(TurbulentCorrelation):

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 0.00128 + 0.1143 * np.maximum(reynolds, 4000) ** (-0.311)
    

class EneaHtsCicc(TurbulentCorrelation):

    def __init__(self, transition_reynolds: float=1e4):
        self.transition_reynolds = transition_reynolds
        self._blasius = Blasius()

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        blasius = self._blasius(reynolds)
        high_reynolds = 2.21 * reynolds ** (-0.4) / 4
        return np.where(
            reynolds <= self.transition_reynolds, 
            blasius,
            high_reynolds
        )
    

class TronzaIterTFBundle(TurbulentCorrelation):
    
    def __init__(self, cross_section: float, hydraulic_diameter: float):
        self.cross_section = cross_section 
        self.hydraulic_diameter = hydraulic_diameter
        _, self.corrected_diameter = evaluate_correction_diameter_hole(hydraulic_diameter)

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return (
            2.46
            * (reynolds / self.hydraulic_diameter * 0.25e-3) ** (-0.52)
            * self.hydraulic_diameter
            / 0.25e-3
            * (
                self.cross_section
                / (0.3 * (39.8e-3 ** 2 - self.corrected_diameter ** 2) * 0.25 * np.pi)
            )
            ** 2
            / 4.0
        )


class LhcPoncetBundle(TurbulentCorrelation):

    def __init__(self):
        self.reynolds_transition_1 = 2e3
        self.reynolds_transition_2 = 4e3
        self.reynolds_transition_3 = 3e4


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        friction_factor = np.zeros_like(reynolds)

        transition = (
            (reynolds >= self.reynolds_transition_1)
            & (reynolds < self.reynolds_transition_2)
        )

        turbulent_blasius = (
            (reynolds >= self.reynolds_transition_2)
            & (reynolds < self.reynolds_transition_3)
        )

        turbulent_petukhov = (
            reynolds >= self.reynolds_transition_3
        )

        friction_factor[transition] = (
            2.6471e-3
            * reynolds[transition] ** 0.3279
            / 4.0
        )

        friction_factor[turbulent_blasius] = (
            0.3194
            * reynolds[turbulent_blasius] ** (-0.25)
            / 4.0
        )

        friction_factor[turbulent_petukhov] = (
            1.0
            / (
                1.8 * np.log10(reynolds[turbulent_petukhov])
                - 1.64
            )**2
            / 4.0
        )

        return friction_factor     
    

class DemoHtsBundle(TurbulentCorrelation):

    def __init__(self):
        self.low_reynolds = 1.5e3
        self.high_reynolds = 2.0e5 


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return np.select(
            [
                reynolds < self.low_reynolds,
                (reynolds >= self.low_reynolds) & (reynolds <= self.high_reynolds),
                reynolds > self.high_reynolds,
            ],
            [
                47.65 * reynolds**(-0.885) / 4.0,
                1.093 * reynolds**(-0.338) / 4.0,
                np.full_like(reynolds, 0.0377 / 4.0),
            ],
        )      


class HtsClBundle(TurbulentCorrelation):

    def __init__(self):
        self.low_reynolds = 1.0e3
        self.high_reynolds = 2.0e3 


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return np.select(
            [
                reynolds < self.low_reynolds,
                (reynolds >= self.low_reynolds) & (reynolds < self.high_reynolds),
                reynolds >= self.high_reynolds,
            ],
            [
                194.002 * reynolds**(-0.52) / 4.0,
                (4.8563 + 4.87e-4 * reynolds) / 4.0,
                14.5148 * reynolds**(-0.12) / 4.0,
            ],
        )
    

class UserDefinedTurbulent(TurbulentCorrelation):
    """
    User hook for custom correlations. The user-defined friction factor is
    carried entirely by ``UserDefinedTotal`` (as in the original
    implementation), so the turbulent contribution is zero.
    """

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return np.zeros_like(reynolds)


"""
    AUXILIARY FUNCTIONS
"""

class NewtonCorrelationCoefficients:
    """
    Provides the correlation coefficients depending on the chosen type.
    """
    def __init__(self, number: int):
        dict_coeff = {
            1: dict(aa=6.4, bb=0.1717, cc=-0.3428, gg=3.05e-3, tthick1=1e-3),
            2: dict(aa=11.88, bb=0.039, cc=-0.299, gg=3.0e-3, tthick1=1e-3),
            3: dict(aa=11.88, bb=0.039, cc=-0.299, gg=1.5e-3, tthick1=1.5e-3),
            4: dict(aa=11.88, bb=0.039, cc=-0.299, gg=3.05e-3, tthick1=1e-3),
            5: dict(aa=6.4, bb=0.1717, cc=-0.3428, gg=2.0e-3, tthick1=1e-3),
            6: dict(aa=6.4, bb=0.1717, cc=-0.3428, gg=3.05e-3, tthick1=1e-3),
            7: dict(aa=11.88, bb=0.039, cc=-0.299, gg=2.40e-3, tthick1=1e-3),
            8: dict(aa=11.88, bb=0.039, cc=-0.299, gg=5.30e-3, tthick1=1e-3),
        }
        self.aa = dict_coeff[number]["aa"]
        self.bb = dict_coeff[number]["bb"]
        self.cc = dict_coeff[number]["cc"]
        self.gg = dict_coeff[number]["gg"]
        self.tthick1 = dict_coeff[number]["tthick1"]


def evaluate_newton(
        hp: np.ndarray,
        coeff: NewtonCorrelationCoefficients,
        hd2: np.ndarray | float,
        goverh: np.ndarray | float,
        tol: float=1e-2,
        frict_guess: float=1e-2,
        max_iter: int=100
    ) -> None:
        """
        Iteratively evaluate the turbulent Darcy friction factor using a
        fixed-point update derived from a Colebrook-type correlation.

        Starting from an initial friction-factor guess, the method repeatedly
        updates the turbulent friction factor until the maximum relative change
        between successive iterations falls below the specified tolerance.

        Note:
            Despite the function name, this implementation does not use a
            Newton-Raphson method. It performs successive substitution
            (fixed-point iteration).

        Args:
            hp (numpy.ndarray):
                Hydraulic parameter used in the friction-factor correlation.
                Must have the same shape as the friction-factor array.

            coeff (CorrelationCoefficients):
                Correlation coefficients.

            hd2 (numpy.ndarray or float):
                Relative roughness term (or equivalent quantity) used in the
                turbulent friction-factor correlation.

            goverh (numpy.ndarray or float):
                Dimensionless geometric parameter used by the correlation.

            tol (float, optional):
                Convergence tolerance based on the maximum relative change in
                friction factor between iterations. Defaults to ``1e-2``.

            frict_guess (float, optional):
                Initial guess for the turbulent friction factor. Defaults to
                ``1e-2``.

            max_iter (int, optional):
                maximum iterations
        """
        turbulent_friction_factor = frict_guess * np.ones(hp.shape)
        err = 10 * np.ones(hp.shape)
        
        for _ in range(max_iter):
            f_turb_old = turbulent_friction_factor
            hpiu = hp * np.sqrt(0.5 * turbulent_friction_factor)
            rhpiu = (
                coeff.aa * hpiu ** coeff.bb * goverh ** coeff.cc
            )
            turbulent_friction_factor = (
                2.0 / (rhpiu - 2.5 * np.log10(hd2) - 3.75) ** 2
            )
            err = np.fabs(
                (turbulent_friction_factor - f_turb_old)
                / turbulent_friction_factor
            )
            if np.max(err) <= tol:
                break 
        else:
            warnings.warn("Maximum iterations reached. Iterative solver eval_ft_newton did not converge.")
        return turbulent_friction_factor