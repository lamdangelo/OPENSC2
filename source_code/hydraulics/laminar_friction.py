from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from hydraulics.auxiliary_functions import (
    evaluate_correction_diameter_hole
)


class LaminarCorrelation(ABC):
    """Base class for laminar friction factor correlations."""

    @abstractmethod
    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        pass


class SmoothTubeLaminar(LaminarCorrelation):
    """
    Fanning friction factor for a smooth circular tube.

    f = 16 / Re
    """

    MIN_RE = 1e-5

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        result = np.zeros_like(reynolds)

        valid = reynolds >= self.MIN_RE
        result[valid] = 16.0 / reynolds[valid]

        return result


class SmoothTubeLaminarLimited(LaminarCorrelation):
    """
    Uses Re=min(Re,2000) for transition smoothing.
    """

    LIMIT_RE = 2000.0
    MIN_RE = 1e-5

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        result = np.zeros_like(reynolds)

        valid = reynolds >= self.MIN_RE
        result[valid] = 16.0 / np.minimum(
            reynolds[valid],
            self.LIMIT_RE,
        )

        return result


class RectangularDuctLaminar(LaminarCorrelation):
    """
    Laminar friction factor for rectangular ducts.

    Based on the correlation currently implemented in
    rectangular_duct_demo_common_memo_laminar_flow_hole().
    """

    def __init__(self, side1: float, side2: float):
        self.alpha = side2 / side1

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        alpha = self.alpha

        coefficient = (
            24.0
            * (
                1.0
                - 1.3553 * alpha
                + 1.9467 * alpha**2
                - 1.7012 * alpha**3
                + 0.9564 * alpha**4
                - 0.2537 * alpha**5
            )
        )

        return coefficient / np.minimum(reynolds, 2000.0)


class TriangularDuctLaminar(LaminarCorrelation):
    """
    Laminar friction factor for triangular ducts.
    """

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 13.33 / np.minimum(reynolds, 2000.0)
    

class IncroperaRectangularLaminarTFHole(LaminarCorrelation):
    """
    Incropera rectangular correlation for TF hole case.
    """
    
    def __init__(self):
        self.thickness = 2.17e-4 
        self.width = 3.28e-3 
        self.correction_factor = self._correction_factor()
        self.laminar_friction_factor = 96.0 
        

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        corrected_reynolds = self.correction_factor * reynolds 
        return self.laminar_friction_factor / corrected_reynolds / 4


    def _correction_factor(self) -> float:
        thickness_width_ratio = self.thickness / self.width
        return 2/3 + 11/24 * thickness_width_ratio * (2 - thickness_width_ratio)
    

class IncroperaRectangularLaminarCSHole(LaminarCorrelation):
    """
    Incropera rectangular correlation for CS hole case.
    """

    def __init__(self, is_rectangular: bool, width: float=None, height: float=None, 
                 hydraulic_diameter: float=None, tthick: float=5e-4):
        if is_rectangular:
            self.side1 = width 
            self.side2 = height
        else:
            self.side1 = tthick
            self.side2 = evaluate_correction_diameter_hole(hydraulic_diameter, tthick)

        self.correction_factor = self._correction_factor()
        self.laminar_friction_factor = self._laminar_friction_factor()


    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        corrected_reynolds = self.correction_factor * reynolds 
        return self.laminar_friction_factor / corrected_reynolds / 4


    def _correction_factor(self) -> float: 
        beta = min(self.side1, self.side2) / max(self.side1, self.side2)
        return 2/3 + 11/24 * beta * (2 - beta)
    

    def _laminar_friction_factor(self) -> float:
        aspect_ratio = max(self.side1, self.side2) / min(self.side1, self.side2)
        aspect_ratios = np.array(
            [1.0, 1.43, 2.0, 3.0, 4.0, 8.0, 9.0]
        )
        poiseuille_numbers = np.array(
            [57.0, 59.0, 62.0, 69.0, 73.0, 82.0, 96.0]
        )
        aspect_ratio = np.clip(
            aspect_ratio,
            aspect_ratios[0],
            aspect_ratios[-1],
        )

        return float(
            np.interp(
                aspect_ratio,
                aspect_ratios,
                poiseuille_numbers,
            )
        )   
    

class DuctDemoCommonTriangular(LaminarCorrelation):
        
    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return 13.33 / np.minimum(reynolds, 2000)
    

class UserDefinedLaminar(LaminarCorrelation):
    """
    User hook for custom correlations. The user-defined friction factor is
    carried entirely by ``UserDefinedTotal`` (as in the original
    implementation), so the laminar contribution is zero.
    """

    def __call__(self, reynolds: np.ndarray) -> np.ndarray:
        return np.zeros_like(reynolds)