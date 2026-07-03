"""
This module owns the fluid-fluid energy transport coefficients K', K'' and
K''' that appear in the source terms of the fluid components' temperature
(and pressure) equations, evaluated at open fluid-fluid interfaces.

Relocated from ``utility_functions/step_matrix_construction.py`` as part of
consolidating the thermal calculations into the ``thermal`` package.
"""

import numpy as np

from components.fluid.fluid_component import FluidComponent
from conductor.conductor import Conductor


def build_transport_coefficients(conductor: Conductor) -> Conductor:
    """Function that builds the transport coefficients K', K'' and K''' that appears in the source terms of the fluid components equations.

    Args:
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        Conductor: updated version of the conductor object.
    """

    key_names = {"K1","K2","K3"}
    for key in key_names:
        # Dictionaties declaration.
        setattr(conductor.gauss_fields, key, dict())

    # Loop in fluid-fluid interfaces.
    # The loop is done on all the type of interfaces between fluids since in
    # the evaluation of K', K'' and K''' the relevant parameter is the open
    # cross section per unit length (open contact perimeter): K', K'' and K'''
    # are not 0 only if the open contact perimeter is not 0. In this way, a
    # check on the interface kind between fluids is avoided for each interface
    # and for each time step.
    for interface in conductor.interface.fluid_fluid:
        comp_1_pressure = interface.comp_1.coolant.gauss_fields.pressure
        comp_2_pressure = interface.comp_2.coolant.gauss_fields.pressure

        # constuct recurrent coefficients of matrix S elements.
        for key in key_names:
            # Initialization.
            getattr(conductor.gauss_fields, key)[interface.interf_name] = np.zeros_like(
                conductor.mesh.gauss_point_coordinates
            )
        # Evaluate pressure difference bethween comp_1 and comp_2
        delta_p = np.abs(comp_1_pressure - comp_2_pressure)
        # Array smart
        delta_p[delta_p < conductor.Delta_p_min] = conductor.Delta_p_min

        # Find index such that # P_comp_2 < P_comp_1.
        ind_1 = np.nonzero(comp_2_pressure < comp_1_pressure)[0]
        # Find index such that P_comp_2 >= P_comp_1.
        ind_2 = np.nonzero(comp_2_pressure >= comp_1_pressure)[0]

        # Compute transport coefficients K', K'' and K'''
        conductor = eval_transport_coefficients(
            conductor,
            interface.interf_name,
            interface.comp_1,
            ind_1, # P_comp_2 < P_comp_1
            delta_p
        )
        conductor = eval_transport_coefficients(
            conductor,
            interface.interf_name,
            interface.comp_2,
            ind_2, # P_comp_2 >= P_comp_1
            delta_p
        )

    return conductor

def eval_transport_coefficients(conductor: Conductor,
    interf_name:str,
    comp:FluidComponent,
    index:np.ndarray,
    delta_p:np.ndarray
    )->Conductor:
    """Function that evaluates the transport coefficients K', K'' and K''' that appears in the source terms of the fluid components equations.

    Args:
        conductor (Conductor): object with all the information of the conductor.
        interf_name (str): name of the interface between fluid component objects.
        comp (FluidComponent): fluid component object of the interface with the dominant pressure (index of the gauss points where this is true are passed in inupt argument index.)
        index (np.ndarray): array with the index of the Gauss points where comp pressure is the dominant one (with respect to the pressure of the other fluid component in the interface)
        delta_p (np.ndarray): array with the pressure differece between the component of the interface).

    Returns:
        Conductor: conductor with updated values of K', K'' and K'''.
    """

    # Aliases
    velocity = comp.coolant.gauss_fields.velocity[index]
    interf_peri = conductor.dict_interf_peri["ch_ch"]["Open"]["Gauss"][
        interf_name
    ]

    # K' evaluation [ms]:
    # K' = A_othogonal*sqrt(2*density/k_loc*abs(Delta_p))
    K1 = (
        interf_peri[index]
        * np.sqrt(
            2.
            * comp.coolant.gauss_fields.total_density[index]
            / (conductor.k_loc * delta_p[index])
        )
    )

    # K'' evaluation [m^2]:
    # K'' = K'*lambda_v*velocity
    K2 = K1 * conductor.lambda_v * velocity

    # K''' evaluation [m^3/s]:
    # K''' = K'*(enthalpy + (velocity*lambda_v)^2/2)
    K3 = (
        K1
        * (
            comp.coolant.gauss_fields.total_enthalpy[index]
            + .5 * (velocity * conductor.lambda_v) ** 2.
        )
    )

    # Assing evaluated K1, K2 and K3 to correspondig key in conductor attribute
    # gauss_fields.
    for key, value in zip(("K1","K2","K3"),(K1,K2,K3)):
        getattr(conductor.gauss_fields, key)[interf_name][index] = value

    return conductor
