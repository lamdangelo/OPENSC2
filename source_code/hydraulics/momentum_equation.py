"""
This module owns the velocity (mass) and pressure (momentum) equation
contributions to the coupled thermal-hydraulic system matrices: the fluid
source Jacobian terms, both for a single fluid component
(:func:`build_smat_fluid_momentum`) and at fluid-fluid interfaces
(:func:`build_smat_fluid_interface_momentum`). The counterpart temperature
row is built by :mod:`thermal.energy_equation`.

Relocated from ``utility_functions/step_matrix_construction.py`` as part of
consolidating the thermal calculations into the ``thermal`` package (the
matching split moved the temperature row out of these functions).
"""

from typing import NamedTuple

import numpy as np

from components.fluid.fluid_component import FluidComponent
from conductor.conductor import Conductor


def build_smat_fluid_momentum(
    matrix:np.ndarray,
    f_comp:FluidComponent,
    eq_idx:NamedTuple,
    )->np.ndarray:

    """Function that builds the velocity and pressure rows of the S matrix (SMAT) therms of the fluid at the Gauss point (SOURCE JACOBIAN).

    N.B. must be called before :func:`thermal.energy_equation.build_smat_fluid_energy`
    on the same matrix, since the latter reuses the friction-factor diagonal
    term computed here (``matrix[eq_idx.velocity, eq_idx.velocity]``).

    Args:
        matrix (np.ndarray): initialized S matrix (np.zeros)
        f_comp (FluidComponent): fluid component object from which get all info to build the coefficients.
        eq_idx (NamedTuple): collection of fluid equation index (velocity, pressure and temperaure equations).

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Fluid velocity at every Gauss point (shallow copy).
    velocity = f_comp.coolant.gauss_fields.velocity
    # velocity equation: main diagonal elements construction
    # (j,j) [vel_j]
    matrix[:, eq_idx.velocity, eq_idx.velocity] = (
        2.0
        # friction_factors[False].total: total friction factor in Gauss
        # points (see __init__ of class Channel for details).
        * f_comp.channel.friction_factors[False].total
        * np.abs(velocity) / f_comp.channel.inputs.hydraulic_diameter
    )

    # pressure equation: elements below main diagonal construction
    # (j+num_fluid_components,0:num_fluid_components) [Pres]
    matrix[:, eq_idx.pressure, eq_idx.velocity] = (
        - matrix[:, eq_idx.velocity, eq_idx.velocity]
        * f_comp.coolant.gauss_fields.Gruneisen
        * f_comp.coolant.gauss_fields.total_density
        * velocity
    )

    return matrix

def build_smat_fluid_interface_momentum(
    matrix:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:

    """Function that builds the velocity and pressure rows of the S matrix (SMAT) therms due to fluid component interfaces at the Gauss point (SOURCE JACOBIAN).

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_momentum.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # NOMENCLATURE
    # h: heat transfer coefficient (_o: open; _c:close)
    # P: contact perimeter (_o: open; _c:close)

    # Alias
    # Collection of NamedTuple with fluid equation index (velocity, pressure
    # and temperaure equations).
    eq_idx = conductor.equation_index

    for interface in conductor.interface.fluid_fluid:

        # Aliases
        K1 = conductor.gauss_fields.K1[interface.interf_name]
        K2 = conductor.gauss_fields.K2[interface.interf_name]
        K3 = conductor.gauss_fields.K3[interface.interf_name]
        interf_peri = conductor.dict_interf_peri["ch_ch"]
        htc_gauss = conductor.gauss_fields.HTC["ch_ch"]

        # coef_htc = P_o * h_o + P_c * h_c
        coef_htc = (
            interf_peri["Open"]["Gauss"][interface.interf_name]
            * htc_gauss["Open"][interface.interf_name]
            + interf_peri["Close"]["Gauss"][interface.interf_name]
            * htc_gauss["Close"][interface.interf_name]
        )

        # Fill rows of comp_1, columns involving comp_1 and comp_2.
        matrix = __smat_fluid_interface_momentum(
            matrix,
            interface.comp_1,
            interface.comp_2,
            eq_idx,
            K1=K1,
            K2=K2,
            K3=K3,
            coef_htc=coef_htc,
        )
        # Fill rows of comp_2, columns involving comp_2 and comp_1.
        matrix = __smat_fluid_interface_momentum(
            matrix,
            interface.comp_2,
            interface.comp_1,
            eq_idx,
            K1=K1,
            K2=K2,
            K3=K3,
            coef_htc=coef_htc,
        )

    return matrix

def __smat_fluid_interface_momentum(
    matrix:np.ndarray,
    comp_1:FluidComponent,
    comp_2:FluidComponent,
    eq_idx:dict,
    **kwargs
    )->np.ndarray:
    """Function that evaluates the velocity and pressure rows of the S matrix (SMAT) terms due to fluid component interfaces at the Gauss point (SOURCE JACOBIAN) by rows (horizontally) corresponding to the equations of comp_1 for the columns of comp_1 including the contributions given by comp_2.

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_momentum.
        comp_1 (FluidComponent): fluid component object from which get all info to build the coefficients.
        comp_2 (FluidComponent): fluid component object from which get all info to build the coefficients.
        eq_idx (dict): collection of NamedTuple with fluid equation index (velocity, pressure and temperaure equations).

    Kwargs:
        K1 (np.ndarray): transport coefficient K'.
        K2 (np.ndarray): transport coefficient K''.
        K3 (np.ndarray): transport coefficient K'''.
        coef_htc (float): heat transfer coefficient per unit of length evauated as the sum of the heat transfer coefficient of the open interface time the corresponding contact perimeter and the heat transfer coefficient of the close intefrace and the corresponding contact perimeter
            P_o * h_o + P_c * h_c.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # NOMENCLATURE
    # w: enthalpy
    # phi: Gruneisen
    # rho: density
    # c0: speed of sound
    # h: heat transfer coefficient (_o: open; _c:close)
    # P: contact perimeter (_o: open; _c:close)
    # A: cross section
    # v: velocity
    # T: temperature
    # c_v: isochoric specific heat

    # Alias
    comp_1_v = comp_1.coolant.gauss_fields.velocity
    comp_1_rho = comp_1.coolant.gauss_fields.total_density
    comp_1_A = comp_1.channel.inputs.cross_section
    comp_1_phi = comp_1.coolant.gauss_fields.Gruneisen
    comp_1_enthalpy = comp_1.coolant.gauss_fields.total_enthalpy
    K1 = kwargs["K1"]
    K2 = kwargs["K2"]
    K3 = kwargs["K3"]
    coef_htc = kwargs["coef_htc"]

    # VELOCITY EQUATION: above/below main diagonal elements construction:
    # (j,j+num_fluid_components) [Pres_j]

    # s_vj_pj = (K1 * v - K2) / (A * rho)

    s_vj_pj = (K1 * comp_1_v - K2) / (comp_1_A * comp_1_rho)

    matrix[
        :,
        eq_idx[comp_1.identifier].velocity,
        eq_idx[comp_1.identifier].pressure,
    ] -= s_vj_pj

    # (j,k + num_fluid_components:2*num_fluid_components)
    # [Pres_k]
    matrix[
        :,
        eq_idx[comp_1.identifier].velocity,
        eq_idx[comp_2.identifier].pressure,
    ] = s_vj_pj

    # PRESSURE EQUATION: main diagonal elements construction:
    # (j+num_fluid_components,j+num_fluid_components) [Pres_j]

    # coef_grun_area = phi / A
    coef_grun_area = comp_1_phi / comp_1_A

    # s_pj_pj = phi/A * [K3 - vK2 - (w - v^2/2 - c0^2/phi)K1]
    #         = coef_grun_area * [K3 - vK2 - (w - v^2/2 - c0^2/phi)K1]
    s_pj_pj = (
        coef_grun_area
        * (K3 - comp_1_v * K2 - (comp_1_enthalpy - comp_1_v ** 2. / 2.
        - comp_1.coolant.gauss_fields.total_speed_of_sound ** 2. / comp_1_phi) * K1
        )
    )

    matrix[
        :,
        eq_idx[comp_1.identifier].pressure,
        eq_idx[comp_1.identifier].pressure,
    ] += s_pj_pj

    # PRESSURE EQUATION: above/below main diagonal elements construction:
    # (j+num_fluid_components,\
    # k + num_fluid_components:2*num_fluid_components) [Pres_k]
    matrix[
        :,
        eq_idx[comp_1.identifier].pressure,
        eq_idx[comp_2.identifier].pressure,
    ] = - s_pj_pj

    # (j+num_fluid_components,j+2*num_fluid_components)
    # [Temp_j] I
    # s_pj_tc = phi/A * (P_o * h_o + P_c * h_c)
    #         = coef_frun_area * coef_htc
    s_pj_tj = coef_grun_area * coef_htc

    matrix[
        :,
        eq_idx[comp_1.identifier].pressure,
        eq_idx[comp_1.identifier].temperature,
    ] += s_pj_tj

    # (j+num_fluid_components,
    # k + 2*num_fluid_components:dict_N_equation
    # ["FluidComponent"]) [Temp_j]
    matrix[
        :,
        eq_idx[comp_1.identifier].pressure,
        eq_idx[comp_2.identifier].temperature,
    ] = - s_pj_tj

    return matrix
