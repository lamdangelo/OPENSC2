"""
This module contains Nusselt correlations.
"""

from abc import ABC, abstractmethod 
from dataclasses import dataclass
import numpy as np 


class NusseltCorrelation(ABC):

    @abstractmethod
    def __call__(self, reynolds: np.ndarray, prandtl: np.ndarray) -> np.ndarray:
        pass 


@dataclass(slots=True, frozen=True)
class DittusBoelter(NusseltCorrelation):

    exponent: float = 0.3

    def __call__(self, reynolds: np.ndarray, prandtl: np.ndarray) -> np.ndarray:
        return 0.023 * reynolds ** 0.8 * prandtl ** self.exponent 
    

class LowerBoundNusselt(NusseltCorrelation):

    def __init__(self, lower_limit: float):
        self.lower_limit = lower_limit 
        self._dittus_boelter = DittusBoelter()

    def __call__(self, reynolds: np.ndarray, prandtl: np.ndarray) -> np.ndarray:
        nu = self._dittus_boelter(reynolds, prandtl)
        return np.maximum(nu, self.lower_limit)
    

class LowerBoundNusselt_119(LowerBoundNusselt):

    def __init__(self, hydraulic_diameter: float, wall_thickness: float=1e-3):
        lower_limit = self._evaluate_lower_limit(hydraulic_diameter, wall_thickness)
        super().__init__(lower_limit=lower_limit)


    def _evaluate_lower_limit(self, hydraulic_diameter: float, 
                              wall_thickness: float) -> float:
        alpha = (
            hydraulic_diameter
            / (hydraulic_diameter + 2.0 * wall_thickness)
        )

        nu_temp = (
            7.541
            * (
                1.0
                - 2.610 * alpha
                + 4.970 * alpha**2
                - 5.119 * alpha**3
                + 2.702 * alpha**4
                - 0.548 * alpha**5
            )
        )

        nu_flux = (
            8.235
            * (
                1.0
                - 10.6044 * alpha
                + 61.1755 * alpha**2
                - 155.1803 * alpha**3
                + 176.9203 * alpha**4
                - 72.9236 * alpha**5
            )
        )

        return 0.5 * (nu_temp + nu_flux)
    

class Correlation_211(NusseltCorrelation):

    def __call__(self, reynolds: np.ndarray, prandtl: np.ndarray=None) -> np.ndarray:
        return np.select(
            [
                reynolds < 1e3,
                (reynolds >= 1e3) & (reynolds < 2e3),
                reynolds >= 2e3,
            ],
            [
                5.0969 * reynolds**0.10,
                10.2029 - 3.3242e-5 * reynolds,
                0.0395 * reynolds**0.73,
            ],
        )    
    

class RectangularDuctEneaHtsCicc(NusseltCorrelation):

    def __init__(self, exponent: float=0.71):
        self.exponent = exponent 

    
    def __call__(self, reynolds: np.ndarray, prandtl: np.ndarray=None) -> np.ndarray:
        return 0.42 * reynolds ** self.exponent 