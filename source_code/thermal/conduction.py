"""
This module owns the solid-component conduction equation builders: the mass
and capacity matrix (heat capacity) and the diffusion matrix (thermal
conduction), evaluated at the Gauss point.

Relocated from ``utility_functions/step_matrix_construction.py`` as part of
consolidating the thermal calculations into the ``thermal`` package.
"""

import numpy as np

from components.solid.solid_component import SolidComponent


def build_mmat_solid(
    matrix:np.ndarray,
    s_comp:SolidComponent,
    eq_idx:int,
    )->np.ndarray:

    """Function that updates the M matrix (MMAT) at the Gauss point, for the SolidComponent equation.

    Args:
        matrix (np.ndarray): M matrix with the element from the fluid equations.
        s_comp (SolidComponent): solid component object from which get all info to build the coefficients.
        eq_idx (int): solid component equation index.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # FORM THE M MATRIX AT EVERY GAUSS POINT (MASS AND CAPACITY)
    # SolidComponent (homogenized) equation.
    # A * rho *cp / cos(theta)
    matrix[:, eq_idx, eq_idx] = (
        s_comp.inputs.cross_section
        * s_comp.gauss_fields.total_density
        * s_comp.gauss_fields.total_isobaric_specific_heat
        / s_comp.inputs.cos_theta
    )

    return matrix

def build_kmat_solid(
    matrix:np.ndarray,
    s_comp:SolidComponent,
    eq_idx:int,
    )->np.ndarray:

    """Function that updates the K matrix (KMAT) at the Gauss point, for the SolidComponent equation.

    Args:
        matrix (np.ndarray): K matrix after call to build_kmat_fluid.
        s_comp (SolidComponent): solid component object from which get all info to build the coefficients.
        eq_idx (int): solid component equation index.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # FORM THE K MATRIX AT EVERY GAUSS POINT (INCLUDING UPWIND)
    # A_{s_comp}*k_{s_comp,homo}; homo = homogenized
    matrix[:, eq_idx, eq_idx] = (
        s_comp.inputs.cross_section
        * s_comp.gauss_fields.total_thermal_conductivity
        / s_comp.inputs.cos_theta
    )

    return matrix
