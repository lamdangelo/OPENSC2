"""
This module contains provides all computations for hydraulics.
"""

import numpy as np
import warnings

from hydraulics.total_friction import TotalFrictionFactor


def compute_gruneisen(isobaric_expansion_coefficient: np.ndarray, 
                      isothermal_compressibility: np.ndarray, 
                      isochoric_specific_heat: np.ndarray,
                      density: np.ndarray) -> np.ndarray:
    """
    Computes the Gruneisen number.
    """
    return (
        isobaric_expansion_coefficient / 
        (isothermal_compressibility * isochoric_specific_heat * density)
    )


def compute_mass_flow_rate(cross_section: float, 
                            velocity: np.ndarray, 
                            density: np.ndarray) -> np.ndarray:
    """
    Computes the mass flow rate.
    """
    return cross_section * velocity * density 


def compute_reynolds_from_mass_flow_rate(hydraulic_diameter: float,
                                         cross_section: float, 
                                         mass_flow_rate: np.ndarray, 
                                         viscosity: np.ndarray) -> np.ndarray:
    """
    Evaluate Reynolds number.
    """
    return (
        mass_flow_rate * hydraulic_diameter 
            / (viscosity * cross_section)
        )



def compute_reynolds_from_velocity(hydraulic_diameter: float,
                                   velocity: np.ndarray,
                                   density: np.ndarray,
                                   viscosity: np.ndarray) -> np.ndarray:
    """
    Computes the Reynolds number from velocity, density and viscosity.
    """
    return np.abs(hydraulic_diameter * velocity * density / viscosity)


def compute_mass_flow_rate_with_direction(flow_sign: int,
                                          cross_section: float,
                                          density: np.ndarray,
                                          velocity: np.ndarray) -> np.ndarray:
    """
    Computes the signed mass flow rate from a flow direction sign, cross-section,
    density, and velocity magnitude.
    """
    return flow_sign * cross_section * density * velocity


def solve_darcy_weisbach_velocity(
    length: float,
    cos_theta: float,
    hydraulic_diameter: float,
    pressure_drop: float,
    density: float,
    viscosity: float,
    max_iterations: int,
    tolerance: float,
    total_friction_factor: TotalFrictionFactor,
    velocity_guess: float = 0.0,
    friction_guess: float = 0.01,
    identifier: str = "",
) -> float:
    """
    Iteratively solves for the steady-state velocity given a prescribed pressure drop
    using an inverted Darcy-Weisbach equation:

        v = sqrt( D_h * dP / (2 * f * rho * L_eff) )

    where L_eff = length / cos_theta is the effective hydraulic length corrected for
    channel inclination. Reynolds number and total friction factor are updated at each
    iteration until the velocity converges.

    Parameters
    ----------
    length : float
        Physical channel length along the conductor axis (m).
    cos_theta : float
        Cosine of the channel inclination angle with respect to the conductor axis.
        Used to compute the effective hydraulic length L_eff = length / cos_theta.
    hydraulic_diameter : float
        Hydraulic diameter of the channel cross-section (m).
    pressure_drop : float
        Prescribed pressure drop across the channel (Pa). Must be positive.
    density : float
        Reference fluid mass density used throughout the iteration (kg/m³).
    viscosity : float
        Reference fluid dynamic viscosity used for Reynolds number evaluation (Pa·s).
    max_iterations : int
        Maximum number of fixed-point iterations before issuing a convergence warning.
    tolerance : float
        Relative convergence criterion on velocity: |v - v_old| / |v| < tolerance.
    total_friction_factor : TotalFrictionFactor (callable)
        total friction factor; a callable that takes Reynolds number as input and
        returns the friction factor
    velocity_guess : float, optional
        Initial velocity estimate to seed the iteration (m/s). Default is 0.0.
    friction_guess : float, optional
        Initial total friction factor estimate to seed the iteration. Default is 0.01.
    identifier : str, optional
        Label used in the convergence warning message to identify the channel.
        Default is an empty string.

    Returns
    -------
    float
        Converged steady-state velocity (m/s). If convergence is not achieved within
        max_iterations, the last iterate is returned and a warning is issued.
    """
    effective_length = length / cos_theta
    total_friction = friction_guess
    velocity = velocity_guess
    err = 1.0
    iter = 1
    while iter <= max_iterations and err >= tolerance:
        iter += 1
        old_velocity = velocity
        velocity = np.sqrt(
            (hydraulic_diameter * pressure_drop)
            / (2.0 * total_friction * density * effective_length)
        )
        reynolds = compute_reynolds_from_velocity(
            hydraulic_diameter, velocity, density, viscosity
        )
        total_friction = total_friction_factor(reynolds)
        err = np.fabs(velocity - old_velocity) / np.fabs(velocity)
    if iter > max_iterations and err >= tolerance:
        warnings.warn(
            f"WARNING: tolerance not achieved for {identifier!r} in GENFLW: err = {err}"
        )
    return velocity