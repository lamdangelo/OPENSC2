"""
This module owns the temperature (energy) equation contributions to the
coupled thermal-hydraulic system matrices and known-term vector:

* the fluid temperature-row source Jacobian terms, both for a single fluid
  component (:func:`build_smat_fluid_energy`) and at fluid-fluid interfaces
  (:func:`build_smat_fluid_interface_energy`) — the counterpart velocity and
  pressure rows are built by
  :mod:`hydraulics.momentum_equation`;
* the fluid-solid convective coupling (:func:`build_smat_fluid_solid_interface`);
* the solid-solid conduction coupling and the solid-environment convective
  coupling (:func:`build_smat_solid_interface`,
  :func:`build_smat_env_solid_interface`);
* the thermal source vector (:func:`build_svec`,
  :func:`build_svec_env_jacket_interface`);
* the known-term vector for the whole thermal-hydraulic linear system
  (:func:`build_known_therm_vector`).

Relocated from ``utility_functions/step_matrix_construction.py`` as part of
consolidating the thermal calculations into the ``thermal`` package. Per the
consolidation plan, functions that mixed fluid velocity/pressure rows with
the fluid temperature row (``build_smat_fluid``, ``build_smat_fluid_interface``,
``__smat_fluid_interface``) were split: the velocity/pressure rows stay in
:mod:`hydraulics.momentum_equation`, and the temperature row moves here. The
row-mixed functions here (``build_smat_fluid_solid_interface``,
``build_smat_solid_interface``, ``build_smat_env_solid_interface``) are moved
wholesale since they exclusively touch fluid/solid temperature dofs.
"""

from typing import NamedTuple

import numpy as np

from components.fluid.fluid_component import FluidComponent
from components.solid.solid_component import SolidComponent
from conductor.conductor import Conductor
from conductor.conductor_flags import HTC_Choice, MethodFlag
from utility_functions.step_matrix_construction import SystemMatrices


def build_smat_fluid_energy(
    matrix:np.ndarray,
    f_comp:FluidComponent,
    eq_idx:NamedTuple,
    )->np.ndarray:
    """Function that builds the temperature-row S matrix (SMAT) therm of the fluid at the Gauss point (SOURCE JACOBIAN).

    N.B. must be called after :func:`hydraulics.momentum_equation.build_smat_fluid_momentum`
    on the same matrix, since it reuses the friction-factor diagonal term
    computed there (``matrix[eq_idx.velocity, eq_idx.velocity]``).

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_momentum.
        f_comp (FluidComponent): fluid component object from which get all info to build the coefficients.
        eq_idx (NamedTuple): collection of fluid equation index (velocity, pressure and temperaure equations).

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Fluid velocity at every Gauss point (shallow copy).
    velocity = f_comp.coolant.gauss_fields.velocity

    # temperature equation: elements below main diagonal construction
    # (j+2*num_fluid_components,0:num_fluid_components) [Temp]
    matrix[:, eq_idx.temperature, eq_idx.velocity] = (
        - matrix[:, eq_idx.velocity, eq_idx.velocity]
        / f_comp.coolant.gauss_fields.total_isochoric_specific_heat
        * velocity
    )

    return matrix

def build_smat_fluid_interface_energy(
    matrix:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:
    """Function that builds the temperature-row S matrix (SMAT) therms due to fluid component interfaces at the Gauss point (SOURCE JACOBIAN).

    Args:
        matrix (np.ndarray): S matrix after call to function hydraulics.momentum_equation.build_smat_fluid_interface_momentum.
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
        matrix = __smat_fluid_interface_energy(
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
        matrix = __smat_fluid_interface_energy(
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

def __smat_fluid_interface_energy(
    matrix:np.ndarray,
    comp_1:FluidComponent,
    comp_2:FluidComponent,
    eq_idx:dict,
    **kwargs
    )->np.ndarray:
    """Function that evaluates the temperature-row S matrix (SMAT) terms due to fluid component interfaces at the Gauss point (SOURCE JACOBIAN) by rows (horizontally) corresponding to the temperature equation of comp_1 for the columns of comp_1 including the contributions given by comp_2.

    Args:
        matrix (np.ndarray): S matrix after call to function hydraulics.momentum_equation.__smat_fluid_interface_momentum.
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
    comp_1_enthalpy = comp_1.coolant.gauss_fields.total_enthalpy
    comp_1_phi = comp_1.coolant.gauss_fields.Gruneisen
    comp_1_cv = comp_1.coolant.gauss_fields.total_isochoric_specific_heat
    K1 = kwargs["K1"]
    K2 = kwargs["K2"]
    K3 = kwargs["K3"]
    coef_htc = kwargs["coef_htc"]

    # TEMPERATURE EQUATION: elements below main diagonal \
    # construction:
    # (j+2*num_fluid_components,j+num_fluid_components) [Pres_j]

    # coef_rho_cv_area = 1/(rho * c_v * A)
    coef_rho_cv_area = 1. / (comp_1_rho * comp_1_cv * comp_1_A)
    # s_tj_pj = 1/(rho * c_v * A) * [K3 - vK2 - (w - v^2/2 - phi*c_v*T)K1]
    #         = coef_rho_cv_area * [K3 - vK2 - (w - v^2/2 - phi*c_v*T)K1]
    s_tj_pj = (
        coef_rho_cv_area
        * (
            K3 - comp_1_v * K2
            - (comp_1_enthalpy - comp_1_v ** 2. / 2.
                - comp_1_phi * comp_1_cv
                * comp_1.coolant.gauss_fields.temperature
            )
            * K1
        )
    )

    matrix[
        :,
        eq_idx[comp_1.identifier].temperature,
        eq_idx[comp_1.identifier].pressure,
    ] += s_tj_pj

    # (j+2*num_fluid_components,\
    # k + num_fluid_components:2*num_fluid_components) [Pres_k]
    matrix[
        :,
        eq_idx[comp_1.identifier].temperature,
        eq_idx[comp_2.identifier].pressure,
    ] = - s_tj_pj

    # TEMPERATURE EQUATION: main diagonal element construction:
    # (j+2*num_fluid_components,j+2*num_fluid_components)
    # [Temp_j] I

    # s_tj_tj = 1/(rho * c_v * A) * (P_o * h_o + P_c * h_c)
    #         = coef_rho_cv_area * coef_htc
    s_tj_tj = coef_rho_cv_area * coef_htc

    matrix[
        :,
        eq_idx[comp_1.identifier].temperature,
        eq_idx[comp_1.identifier].temperature,
    ] += s_tj_tj

    # TEMPERATURE EQUATION: above/below main diagonal elements
    # construction:
    # (j+2*num_fluid_components,k + 2*num_fluid_components)
    # [Temp_k]
    matrix[
        :,
        eq_idx[comp_1.identifier].temperature,
        eq_idx[comp_2.identifier].temperature,
    ] = - s_tj_tj

    return matrix

def build_smat_fluid_solid_interface(
    matrix:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:

    """Function that builds the S matrix (SMAT) therms due to fluid-solid component interfaces at the Gauss point (SOURCE JACOBIAN).

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_interface_energy (and hydraulics.momentum_equation.build_smat_fluid_interface_momentum).
        conductor (Conductor): object with all the information of the conductor.
        eq_idx (dict): collection of NamedTuple with fluid equation index (velocity, pressure and temperaure equations) and of integer for solid equation index.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # NOMENCLATURE
    # phi: Gruneisen
    # rho: density
    # h: heat transfer coefficient (_o: open; _c:close)
    # P: contact perimeter (_o: open; _c:close)
    # A: cross section
    # c_v: isochoric specific heat

    # Alias
    # Collection of NamedTyple with fluid equation index (velocity, pressure
    # and temperaure equations).
    eq_idx = conductor.equation_index

    for interface in conductor.interface.fluid_solid:

        comp_1_A = interface.comp_1.channel.inputs.cross_section
        # pressure equation: above main diagonal elements construction.
        # (j+num_fluid_components,j+2*num_fluid_components) [Temp_j] II + III

        # coef_grun_area = phi / A
        coef_grun_area = (
            interface.comp_1.coolant.gauss_fields.Gruneisen
            / comp_1_A
        )

        # coef_htc = P * h
        coef_htc = (
            conductor.dict_interf_peri["ch_sol"]["Gauss"][
                interface.interf_name]
            * conductor.gauss_fields.HTC["ch_sol"][interface.interf_name]
        )

        # s_pj_tj = phi / A * P * h
        #         = coef_grun_area * coef_htc
        s_pj_tj = coef_grun_area * coef_htc

        matrix[
            :,
            eq_idx[interface.comp_1.identifier].pressure,
            eq_idx[interface.comp_1.identifier].temperature
        ] += s_pj_tj

        # (j+num_fluid_components,l + equation_counts.fluid_equations) [Temp_l]
        matrix[
            :,
            eq_idx[interface.comp_1.identifier].pressure,
            eq_idx[interface.comp_2.identifier],
        ] = - s_pj_tj

        # temperature equation: main diagonal element construction
        # (j+2*num_fluid_components,j+2*num_fluid_components) [Temp_j] II + III

        # coef_rho_cv_area = 1/(rho * c_v * A)
        coef_rho_cv_area = 1. / (
            interface.comp_1.coolant.gauss_fields.total_density
            * interface.comp_1.coolant.gauss_fields.total_isochoric_specific_heat
            * comp_1_A
        )

        # s_tj_tj = 1/(rho * c_v * A) * P * h
        #         = coef_rho_cv_area * coef_htc
        s_tj_tj = coef_rho_cv_area * coef_htc

        matrix[
            :,
            eq_idx[interface.comp_1.identifier].temperature,
            eq_idx[interface.comp_1.identifier].temperature,
        ] += s_tj_tj

        # temperature equation: above main diagonal elements construction
        # (j+2*num_fluid_components,l + equation_counts.fluid_equations) [Temp_l]
        matrix[
            :,
            eq_idx[interface.comp_1.identifier].temperature,
            eq_idx[interface.comp_2.identifier],
        ] = - s_tj_tj

        # SOLID COMPONENTS CONDUCTION EQUATION: main diagonal element
        # construction.
        # (l + equation_counts.fluid_equations,l + equation_counts.fluid_equations) [Temp_l] I
        matrix[
            :,
            eq_idx[interface.comp_2.identifier],
            eq_idx[interface.comp_2.identifier],
        ] += coef_htc

        # SOLID COMPONENTS CONDUCTION EQUATION: below main diagonal elements
        # construction.
        # (l + equation_counts.fluid_equations,l + 2*num_fluid_components) [Temp_j]
        matrix[
            :,
            eq_idx[interface.comp_2.identifier],
            eq_idx[interface.comp_1.identifier].temperature,
        ] = -coef_htc

    return matrix

def build_smat_solid_interface(
    matrix:np.ndarray,
    conductor:Conductor,
    )->np.ndarray:

    """Function that builds the S matrix (SMAT) therms due to solid component interfaces at the Gauss point (SOURCE JACOBIAN).

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_solid_interface.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # NOMENCLATURE
    # P: contact perimeter
    # h_conv: convective heat transfer coefficient.

    # Alias
    # Collection of integer solid equation index.
    eq_idx = conductor.equation_index

    for interface in conductor.interface.solid_solid:

        # coef_htc = P * h_conv W / m / K
        coef_htc = (
            conductor.dict_interf_peri["sol_sol"]["Gauss"][interface.interf_name]
            * conductor.gauss_fields.HTC["sol_sol"]["cond"][
                interface.interf_name
            ]
        )

        # Fill rows of comp_1, columns involving comp_1 and comp_2.
        matrix = __smat_solid_interface(
            matrix,
            interface.comp_1,
            interface.comp_2,
            eq_idx,
            coef_htc=coef_htc,
        )
        # Fill rows of comp_2, columns involving comp_2 and comp_1.
        matrix = __smat_solid_interface(
            matrix,
            interface.comp_2,
            interface.comp_1,
            eq_idx,
            coef_htc=coef_htc,
        )

    return matrix

def __smat_solid_interface(
    matrix:np.ndarray,
    comp_1:SolidComponent,
    comp_2:SolidComponent,
    eq_idx:dict,
    **kwargs,
    )-> np.ndarray:
    """Function that evaluates the S matrix (SMAT) therms due to solid component interfaces at the Gauss point (SOURCE JACOBIAN) by rows (horizontally) corresponding to the equations of comp_1 for the columns of comp_1 including the contributions given by comp_2.

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_fluid_solid_interface.
        comp_1 (SolidComponent): solid component object from which get all info to build the coefficients.
        comp_2 (SolidComponent): solid component object from which get all info to build the coefficients.
        eq_idx (dict): collection of solid component equation index.

    Kwargs:
        coef_htc (float): heat transfer coefficient per unit of length
            P * h_conv.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    coef_htc = kwargs["coef_htc"]
    # SOLID COMPONENTS CONDUCTION EQUATION: main diagonal element
    # construction:
    # (l + equation_counts.fluid_equations,l
    # + equation_counts.fluid_equations) [Temp_l] II + III
    matrix[
        :,
        eq_idx[comp_1.identifier],
        eq_idx[comp_1.identifier],
        ] += coef_htc

    # SOLID COMPONENTS CONDUCTION EQUATION: above/below main diagonal
    # elements construction:
    # (l + equation_counts.fluid_equations,m
    # + equation_counts.fluid_equations) [Temp_m]
    matrix[
        :,
        eq_idx[comp_1.identifier],
        eq_idx[comp_2.identifier],
    ] = - coef_htc

    return matrix

def build_smat_env_solid_interface(
    matrix:np.ndarray,
    conductor:Conductor,
    interface:NamedTuple,
    )->np.ndarray:

    """Function that builds the S matrix (SMAT) therms due to environment and solid component interfaces at the Gauss point (SOURCE JACOBIAN).

    Args:
        matrix (np.ndarray): S matrix after call to function build_smat_solid_interface.
        conductor (Conductor): object with all the information of the conductor.
        interface (NamedTuple): collection of interface information like interface name and components that constitute the interface.

    Returns:
        np.ndarray: matrix with updated elements.
    """

    # Alias.
    h_conv = conductor.gauss_fields.HTC["env_sol"][
                interface.interf_name
            ]["conv"]
    # Collection of integer solid equation index.
    eq_idx = conductor.equation_index

    # Convective heating with the external environment (implicit treatment).
    if (
        HTC_Choice.get_htc_choice_flag(
            conductor.coupling.htc_choice[
                interface.comp_1.KIND,
                interface.comp_2.identifier,
            ]
        )
        == HTC_Choice.CONVECTION_HTC_COMPUTED
        and conductor.inputs.is_rectangular
    ):
        # Rectangular duct.
        coef_htc = (
            + 2. * conductor.inputs.height * h_conv["side"]
            + conductor.inputs.width
            * (
                h_conv["bottom"] + h_conv["top"]
            )
        )
    else:
        coef_htc = (
            h_conv
            * conductor.dict_interf_peri["env_sol"]["Gauss"][
                interface.interf_name
            ]
        )
    # Update matrix coefficients.
    matrix[
            :,
            eq_idx[interface.comp_2.identifier],
            eq_idx[interface.comp_2.identifier],
        ] += coef_htc

    return matrix

def build_svec(
    array:np.ndarray,
    s_comp: SolidComponent,
    eq_idx:int,
    **kwargs,
    )->np.ndarray:
    """Function that builds the source vector (SVEC) elements at the Gauss point due to heat generation in strand and or jacket component objects and to thermal contact beween jacket compoments belonging to different conductors (qsource). For strand component objects the latter contribution is always zero.
    N.B. This function is a merge of the if statement if isinstance(scomp,StrandComponent) is true do not account for qsource else, account for qsource. Since, as mentioned qsourse = 0 for StrandComponent, the check chan be avoided. This should improve readability and maintainability of the function.

    Args:
        array (np.ndarray): initialized SVEC array.
        s_comp (SolidComponent): solid component object from which get all info to build the coefficients.
        eq_idx (int): solid equation index.

    Kwargs:
        num_step (int): present time step counter value.
        qsource (np.ndarray): matrix with heat due to thermal contact between jacket components of different conductors.
        comp_idx (int): component index, used to correctly assign the heat source term due to thermal contact between solid components of different conductors.

    Returns:
        np.ndarray: array with updated elements.
    """

    # Alias.
    qsource = kwargs["qsource"]
    comp_idx = kwargs["comp_idx"]
    Q1 = s_comp.gauss_fields.Q1
    Q2 = s_comp.gauss_fields.Q2

    # N.B. qsource has non zero values only in nodes and columns that represent
    # the contact between jacket components of different conductors.

    # This is independent from the solution method thanks to the escamotage of
    # the dummy steady state corresponding to the initialization.
    # Number of elements: Q1/Q2 are Gauss-point (element) arrays while
    # qsource is a nodal array.
    number_of_elements = Q1.shape[0]
    qsource_left = qsource[:number_of_elements, comp_idx]
    qsource_right = qsource[1:number_of_elements + 1, comp_idx]

    if kwargs["num_step"] == 1:
        # Present time step.
        array.present[:, eq_idx, 0] = Q1[:, 0] - qsource_left
        array.present[:, eq_idx, 1] = Q2[:, 0] - qsource_right
        # Previous time step.
        array.previous[:, eq_idx, 0] = Q1[:, 1] - qsource_left
        array.previous[:, eq_idx, 1] = Q2[:, 1] - qsource_right
    else:
        # Compute only at the current time step.
        array[:, eq_idx, 0] = Q1[:, 0] - qsource_left
        array[:, eq_idx, 1] = Q2[:, 0] - qsource_right

    return array

def build_svec_env_jacket_interface(
    array:np.ndarray,
    conductor: Conductor,
    interface:NamedTuple,
    )->np.ndarray:
    """Function that builds the source vector (SVEC) terms at the Gauss point due to heat transfer by convection and/or radiation between environment and jacket component objects.

    Args:
        array (np.ndarray): SVEC array after call to function build_svec.
        interface (NamedTuple): collection of interface information like interface name and components that constitute the interface.

    Returns:
        np.ndarray: array with updated elements.
    """

    # Alias.
    h_conv = conductor.gauss_fields.HTC["env_sol"][
                interface.interf_name
            ]["conv"]
    height = conductor.inputs.height
    width = conductor.inputs.width
    env = interface.comp_1
    s_comp = interface.comp_2
    # Collection of integer solid equation index.
    eq_idx = conductor.equation_index

    # Add the contribution of the external heating by convection to the
    # known term vector.
    if (
        HTC_Choice.get_htc_choice_flag(
            conductor.coupling.htc_choice[
                env.KIND, s_comp.identifier
            ]
        )
        == HTC_Choice.CONVECTION_HTC_COMPUTED
        and conductor.inputs.is_rectangular
    ):
        # Rectangular duct.
        # N.B. bug preserved from the legacy per-element implementation: the
        # width contribution was a stray statement with no effect and the
        # heat was only added in the non-rectangular branch.
        coef = 2. * height * h_conv["side"]
        + width* (h_conv["bottom"] + h_conv["top"])
    else:
        coef = (
            conductor.dict_interf_peri["env_sol"]["Gauss"][
                interface.interf_name
            ]
            * h_conv
        )

        # Linear heat flux from environment W/m
        env_heat = coef * env.inputs["Temperature"]

        if conductor.cond_num_step == 1:
            # Present time step.
            array.present[:, eq_idx[s_comp.identifier], 0] += env_heat
            array.present[:, eq_idx[s_comp.identifier], 1] += env_heat
            # Previous time step.
            array.previous[:, eq_idx[s_comp.identifier], 0] += env_heat
            array.previous[:, eq_idx[s_comp.identifier], 1] += env_heat
        else:
            # Present time step.
            array[:, eq_idx[s_comp.identifier], 0] += env_heat
            array[:, eq_idx[s_comp.identifier], 1] += env_heat

    return array

def build_known_therm_vector(
    array:np.ndarray,
    aux_matrices:SystemMatrices,
    conductor:Conductor
)->np.ndarray:
    """Function that builds the known therm vector for the thermal hydraulic problem according to the selected method for time integration.

    Args:
        array (np.ndarray): initialized array Known.
        aux_matrices (SystemMatrices): collection of matrix MASMAT, FLXMAT, DIFMAT and SORMAT after call to function assemble_system_matrices.
        conductor (Conductor): object with all the information of the conductor.

    Returns:
        np.ndarray: array with updated elements.
    """

    # Alias
    total = conductor.equation_counts.total_equations
    half = conductor.band.half_bandwidth
    half_1 = half - 1
    method = conductor.inputs.thermohydraulic_method
    load_vector = conductor.time_integration.load_vector # shallow copy
    solution_history = conductor.time_integration.solution # shallow copy

    if method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Alias
        am4_aa = conductor.time_integration.adams_moulton_matrices # shallow copy
        am4_coef = np.array((9.,19.,5.,- 1.)) / 24.

    # Unpack auxiliary matrices (mass capacity, flux Jacobian, diffusion,
    # source Jacobian).
    mass_capacity, flux_jacobian, diffusion, source_jacobian = aux_matrices

    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson.
        # The known term is the banded matrix-vector product
        #   Known = (M/dt - (1 - theta) * (F + D + S)) @ solution_history[:, 0]
        # in the legacy band layout (storage column c holds matrix row c,
        # entry (c, j) at storage row half_1 + j - c). It is evaluated one
        # band diagonal at a time with slice arithmetic instead of the
        # previous per-row Python loop.
        banded_matrix = mass_capacity / conductor.time_step - (
            1.0 - conductor.theta_method
        ) * (flux_jacobian + diffusion + source_jacobian)
        solution = solution_history[:, 0]
        array[:] = 0.0
        for diagonal in range(-half_1, half_1 + 1):
            first = max(0, -diagonal)
            last = total - max(0, diagonal)
            array[first:last] += (
                banded_matrix[half_1 + diagonal, first:last]
                * solution[first + diagonal : last + diagonal]
            )

    # ADD THE LOAD CONTRIBUTION FROM PREVIOUS STEP
    # c_mat_idx: column index of the auxiliary matrices (MASMAT,FLXMAT,DIFMAT,
    # SORMAT); used also as row index of the known term vector.
    for c_mat_idx in range(total if method == MethodFlag.ADAMS_MOULTON_4TH_ORDER else 0):
        if c_mat_idx <= half_1:
            # remember that arange stops before the stop value:
            # last value = stop - step
            # r_arr_idx: row index of the solution_history array
            r_arr_idx = np.arange(
                start=0,
                stop=half + c_mat_idx,
                step=1,
                dtype=int,
            )
        elif c_mat_idx >= total - half_1:
            r_arr_idx = np.arange(
                start=c_mat_idx - half_1,
                stop=total,
                step=1,
                dtype=int,
            )
        else:
            r_arr_idx = np.arange(
                start=c_mat_idx - half_1,
                stop=c_mat_idx + half,
                step=1,
                dtype=int,
            )
        # r_mat_idx: row index of the auxiliary matrices (MASMAT,FLXMAT,DIFMAT,
        # SORMAT)
        r_mat_idx = r_arr_idx - c_mat_idx + half_1
        if method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
            # Adams-Moulton order 4
            # Matrices vectors product contribution
            # np.sum(am4_coef[2:] * am4_aa[2:,r_mat_idx,c_mat_idx].T * solution_history[r_arr_idx,1:3],1) should be equivalent to 5. / 24.* am4_aa[2,r_mat_idx,c_mat_idx] * solution_history[r_arr_idx, 1] - 1. / 24. * am4_aa[3,r_mat_idx,c_mat_idx] * solution_history[r_arr_idx, 2]
            array[c_mat_idx] = np.sum(
                (
                    mass_capacity[r_mat_idx, c_mat_idx] / conductor.time_step
                    - am4_coef[1] * am4_aa[1,r_mat_idx,c_mat_idx]
                ) * solution_history[r_arr_idx,0] # array of shape (r_arr_idx.shape[0],)
                + np.sum(
                    am4_coef[2:] * am4_aa[2:,r_mat_idx,c_mat_idx].T
                    * solution_history[r_arr_idx,1:3],1
                ) # array of shape (r_arr_idx.shape[0],)
            ) # array of shape (1,)

    if method in (MethodFlag.BACKWARD_EULER, MethodFlag.CRANK_NICOLSON):
        # Backward Euler or Crank-Nicolson
        # External sources (load vector) contribution
        array += (
            + conductor.theta_method * load_vector[:,0]
            + (1.0 - conductor.theta_method) * load_vector[:,1]
        )
    elif method == MethodFlag.ADAMS_MOULTON_4TH_ORDER:
        # Adams-Moulton order 4

        # Chance coefficient sign to exploit sum (array smart).
        am4_coef[2:] = - am4_coef[2:]
        # External sources (load vector) contribution
        array += np.sum(am4_coef * load_vector,1)

    return array
