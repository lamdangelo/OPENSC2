from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Base API
# ---------------------------------------------------------------------------

class TotalFrictionFactor(ABC):
    """
    Combines laminar and turbulent friction factors
    into a single total friction factor.
    """

    @abstractmethod
    def __call__(
        self,
        reynolds: np.ndarray,
        laminar: np.ndarray,
        turbulent: np.ndarray,
    ) -> np.ndarray:
        pass


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class MaximumModel(TotalFrictionFactor):
    """
    Original implementation:

        f_total = max(f_laminar, f_turbulent)

    Used by most correlations.
    """

    def __call__(
        self,
        reynolds: np.ndarray,
        laminar: np.ndarray,
        turbulent: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(laminar, turbulent)


class TransitionalInterpolation(TotalFrictionFactor):
    """
    Linear transition between laminar and turbulent regimes.

    Original implementation:

        Re <= 2000      -> laminar
        2000 < Re < Rt -> interpolation
        Re >= Rt       -> turbulent
    """

    def __init__(
        self,
        transition_start: float = 2000.0,
        transition_end: float = 4000.0,
    ):
        self.transition_start = transition_start
        self.transition_end = transition_end

    def __call__(
        self,
        reynolds: np.ndarray,
        laminar: np.ndarray,
        turbulent: np.ndarray,
    ) -> np.ndarray:

        result = np.empty_like(reynolds)

        re = reynolds

        laminar_region = re <= self.transition_start

        transition_region = (
            (re > self.transition_start)
            & (re < self.transition_end)
        )

        turbulent_region = re >= self.transition_end

        result[laminar_region] = laminar[laminar_region]

        if np.any(transition_region):

            weight = (
                re[transition_region]
                - self.transition_start
            ) / (
                self.transition_end
                - self.transition_start
            )

            result[transition_region] = (
                laminar[transition_region]
                + weight
                * (
                    turbulent[transition_region]
                    - laminar[transition_region]
                )
            )

        result[turbulent_region] = turbulent[turbulent_region]

        return result


class UserDefinedTotal(TotalFrictionFactor):
    """
    User hook for custom implementations.

    The user should write here the friction factor value or correlation (it
    must be array smart if a function of Reynolds). The default is a constant
    1.0; the actual value is set through the friction multiplier, which is
    applied in ``Channel.evaluate_friction_factors``.
    """

    def __call__(
        self,
        reynolds: np.ndarray,
        laminar: np.ndarray,
        turbulent: np.ndarray,
    ) -> np.ndarray:
        return np.ones_like(reynolds)