"""
This module owns the evaluation of heat transfer coefficients (HTC) between
every pair of interface kinds in the conductor: channel-solid, channel-channel,
solid-solid, and environment-solid. It also owns the channel-side transport
property preamble (coolant properties, steady-state HTC, friction factor) that
must run before any of the four HTC evaluations.

Relocated from ``conductor/conductor.py::eval_transp_coeff`` (renamed
``evaluate_transport_coefficients``) as part of splitting the file for single
responsibility. ``Conductor.evaluate_transport_coefficients`` is now a thin
orchestrator that calls :func:`evaluate_channel_transport_properties` followed
by the four ``evaluate_*_htc`` functions below, in that order (the ordering
matters: heat-transfer coefficients are consumed downstream, e.g.
``mass_energy_balance`` reads ``HTC["env_sol"]``).
"""

import os

import numpy as np

from conductor.conductor_flags import HTC_Choice
from properties_of_materials.stainless_steel import thermal_conductivity_ss
import thermal.radiation as thermal_radiation


def evaluate_channel_transport_properties(conductor: object, simulation: object, flag_nodal: bool) -> None:
    """Evaluate coolant properties, steady-state HTC and friction factor for every fluid component.

    Must run before any of the ``evaluate_*_htc`` functions in this module,
    since they read ``fluid_comp.channel.steady_state_htc``.
    """
    for fluid_comp in conductor.inventory.fluids.collection:

        # Evaluate coolant properties in nodal points.
        fluid_comp.coolant._eval_properties_nodal_gauss(
            conductor, simulation.fluid_prop_aliases, flag_nodal
        )

        # Define dictionary to select nodal or gauss properties according to the value of flag_nodal.0
        dict_dummy_chan = {
            True: fluid_comp.coolant.node_fields,
            False: fluid_comp.coolant.gauss_fields,
        }
        # Evaluate steady state heat transfer coefficient for each channel.
        fluid_comp.channel.eval_steady_state_htc(
            dict_dummy_chan[flag_nodal], nodal=flag_nodal
        )
        # Evaluate total friction factor for each channel.
        fluid_comp.channel.eval_friction_factor(
            dict_dummy_chan[flag_nodal].Reynolds, nodal=flag_nodal
        )
    # end for loop fluid_comp.


def evaluate_channel_solid_htc(conductor: object, simulation: object, dict_dummy: object, flag_nodal: bool) -> int:
    """Evaluate the HTC (``dict_dummy.HTC["ch_sol"]``) at every channel-solid interface.

    Combines steady-state convective HTC with a transient Kapitza-resistance
    contribution, or reads a user-specified value from file, depending on the
    interface's :class:`HTC_Choice`.

    Returns:
        int: number of channel-solid interfaces evaluated (for the total
        interface-count validation performed by the caller).
    """
    interf_flag = conductor.coupling.contact_perimeter_flag
    htc_len = 0

    for fluid_comp_r in conductor.inventory.fluids.collection:
        dict_dummy_chan_r = {
            True: fluid_comp_r.coolant.node_fields,
            False: fluid_comp_r.coolant.gauss_fields,
        }
        # Read the submatrix containing information about channel - solid objects iterfaces (cdp, 06/2020)
        # nested loop on channel - solid objects (cpd 06/2020)
        for s_comp in conductor.inventory.solids.collection:
            dict_dummy_comp = {
                True: s_comp.node_fields,
                False: s_comp.gauss_fields,
            }
            # Multiplier used in both cases (positive and negative flag).
            mlt = conductor.coupling.htc_multiplier[
                        fluid_comp_r.identifier, s_comp.identifier
                    ]
            # Rationale: compute dictionary vaules only if there is an interface \
            # (cdp, 09/2020)
            if (
                abs(interf_flag[
                        fluid_comp_r.identifier, s_comp.identifier
                    ]
                ) == 1
            ):
                htc_len = htc_len + 1
                # new channel-solid interface (cdp, 09/2020)
                htc_Kapitza = np.zeros(
                    dict_dummy_chan_r[flag_nodal]["temperature"].shape
                )
                htc_transient = np.zeros(
                    dict_dummy_chan_r[flag_nodal]["temperature"].shape
                )
                htc_full_transient = np.zeros(
                    dict_dummy_chan_r[flag_nodal]["temperature"].shape
                )
                if (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        fluid_comp_r.identifier, s_comp.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_COMPUTED
                ):
                    htc_Kapitza = (
                        200.0
                        * (
                            dict_dummy_comp[flag_nodal]["temperature"]
                            + dict_dummy_chan_r[flag_nodal]["temperature"]
                        )
                        * (
                            dict_dummy_comp[flag_nodal]["temperature"] ** 2
                            + dict_dummy_chan_r[flag_nodal]["temperature"] ** 2
                        )
                    )
                    if conductor.cond_time[-1] > s_comp.operations.heat_flux_time_start:
                        # implementation fully correct only for fully implicit method \
                        # (cdp, 06/2020)
                        htc_transient = np.sqrt(
                            (
                                dict_dummy_chan_r[flag_nodal][
                                    "total_thermal_conductivity"
                                ]
                                * dict_dummy_chan_r[flag_nodal]["total_density"]
                                * dict_dummy_chan_r[flag_nodal][
                                    "total_isobaric_specific_heat"
                                ]
                            )
                            / (
                                np.pi
                                * (conductor.cond_time[-1] - s_comp.operations.heat_flux_time_start)
                            )
                        )
                        htc_full_transient = (htc_Kapitza * htc_transient) / (
                            htc_Kapitza + htc_transient
                        )
                    # Assign to the HTC key of dictionary dict_dummy the dictionary whit the information about heat trasfer coefficient betweent channel fluid_comp_r and solid s_comp. Interface identification is given by the key name itself: f"{fluid_comp_r.identifier}_{s_comp.identifier}". This inner dictionary consists of a single key-value pair. (cdp, 07/2020)
                    dict_dummy.HTC["ch_sol"][
                        conductor.dict_topology["ch_sol"][fluid_comp_r.identifier][
                            s_comp.identifier
                        ]
                    ] = np.maximum(
                        fluid_comp_r.channel.steady_state_htc[flag_nodal] * mlt,
                        htc_full_transient,
                    )
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        fluid_comp_r.identifier, s_comp.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_READ_FROM_FILE
                ):
                    dict_dummy.HTC["ch_sol"][
                        conductor.dict_topology["ch_sol"][fluid_comp_r.identifier][
                            s_comp.identifier
                        ]
                    ] = conductor.coupling.htc_contact[
                        fluid_comp_r.identifier, s_comp.identifier
                    ] * np.ones(
                        dict_dummy_chan_r[flag_nodal]["temperature"].shape
                    ) * mlt
        # end loop on SolidComponent
    # end for loop fluid_comp_r.
    return htc_len


def evaluate_channel_channel_htc(conductor: object, simulation: object, dict_dummy: object, flag_nodal: bool) -> int:
    """Evaluate the HTC (``dict_dummy.HTC["ch_ch"]["Open"]``/``["Close"]``) at every channel-channel interface.

    "Open" is the interface without a separating wall (harmonic mean of the
    two channels' HTC); "Close" includes the separating-wall thermal
    resistance.

    Returns:
        int: number of channel-channel interfaces evaluated.
    """
    interf_flag = conductor.coupling.contact_perimeter_flag
    htc_len = 0

    for rr, fluid_comp_r in enumerate(conductor.inventory.fluids.collection):
        dict_dummy_chan_r = {
            True: fluid_comp_r.coolant.node_fields,
            False: fluid_comp_r.coolant.gauss_fields,
        }
        # nested loop on channel - channel objects (cdp, 06/2020)
        for _, fluid_comp_c in enumerate(
            conductor.inventory.fluids.collection[rr + 1 :]
        ):
            dict_dummy_chan_c = {
                True: fluid_comp_c.coolant.node_fields,
                False: fluid_comp_c.coolant.gauss_fields,
            }

            # Multiplier used in both cases (positive and negative flag).
            mlt = conductor.coupling.htc_multiplier[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ]
            if (
                abs(interf_flag[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ]
                ) == 1
            ):
                # new channel-channel interface (cdp, 09/2020)
                htc_len = htc_len + 1
                # Construct interface name: it can be found also in dict_topology["ch_ch"] but a search in dictionaties "Hydraulic_parallel" and "Thermal_contact" should be performed, which makes thinks not easy to do; it is simpler to construct interface names combining channels identifier (cdp, 09/2020)
                interface_name = (
                    f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                )
                if (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_COMPUTED
                ):
                    # dummy
                    htc1 = fluid_comp_r.channel.steady_state_htc[flag_nodal]
                    # dummy
                    htc2 = fluid_comp_c.channel.steady_state_htc[flag_nodal]

                    dict_dummy.HTC["ch_ch"]["Open"][interface_name] = (
                        mlt * htc1 * htc2 / (htc1 + htc2)
                    )
                    cond_interface = thermal_conductivity_ss(
                        (
                            dict_dummy_chan_r[flag_nodal]["temperature"]
                            + dict_dummy_chan_c[flag_nodal]["temperature"]
                        )
                        / 2
                    )
                    R_wall = (
                        conductor.coupling.interface_thickness[
                            fluid_comp_r.identifier, fluid_comp_c.identifier
                        ]
                        / cond_interface
                    )
                    dict_dummy.HTC["ch_ch"]["Close"][interface_name] = mlt / (
                        1 / htc1 + 1 / htc2 + R_wall
                    )
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_READ_FROM_FILE
                ):
                    # in this case it is assumed that both open and close hct have the same value (cdp, 07/2020)
                    dict_dummy.HTC["ch_ch"]["Open"][
                        interface_name
                    ] = conductor.coupling.htc_contact[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ] * np.ones(
                        dict_dummy_chan_r[flag_nodal]["temperature"].shape
                    ) * mlt
                    dict_dummy.HTC["ch_ch"]["Close"][
                        interface_name
                    ] = conductor.coupling.htc_contact[
                        fluid_comp_r.identifier, fluid_comp_c.identifier
                    ] * np.ones(
                        dict_dummy_chan_r[flag_nodal]["temperature"].shape
                    ) * mlt
        # end for loop cc
    # end for loop rr
    return htc_len


def evaluate_solid_solid_htc(conductor: object, simulation: object, dict_dummy: object, flag_nodal: bool) -> int:
    """Evaluate the HTC (``dict_dummy.HTC["sol_sol"]["cond"]``/``["rad"]``) at every solid-solid interface.

    "cond" is the conductive HTC through direct contact (thermal-resistance
    network of the two components plus contact resistance); "rad" is the
    radiative HTC between the two surfaces.

    Note: the radiative branch may overwrite
    ``conductor.coupling.contact_perimeter`` with the smaller of
    the two components' perimeters when they are geometrically nested (inner
    convex / outer concave surfaces) — this write-back is preserved exactly
    as in the original monolithic method; do not turn this into a pure
    function that silently drops the mutation.

    Returns:
        int: number of solid-solid interfaces evaluated.
    """
    interf_flag = conductor.coupling.contact_perimeter_flag
    htc_len = 0

    for rr, s_comp_r in enumerate(conductor.inventory.solids.collection):
        dict_dummy_comp_r = {
            True: s_comp_r.node_fields,
            False: s_comp_r.gauss_fields,
        }
        # Thermal conductivity of s_comp_c
        kk_s_comp_r = dict_dummy_comp_r[flag_nodal][
            "total_thermal_conductivity"
        ] # W/m/K
        for _, s_comp_c in enumerate(
            conductor.inventory.solids.collection[rr + 1 :]
        ):
            dict_dummy_comp_c = {
                True: s_comp_c.node_fields,
                False: s_comp_c.gauss_fields,
            }
            # Multiplier used in both cases (positive and negative flag).
            mlt = conductor.coupling.htc_multiplier[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
            if (
                abs(interf_flag[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                ) == 1
            ):
                dict_dummy.HTC["sol_sol"]["cond"][
                    conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                        s_comp_c.identifier
                    ]
                ] = np.zeros(dict_dummy_comp_r[flag_nodal]["temperature"].shape)
                dict_dummy.HTC["sol_sol"]["rad"][
                    conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                        s_comp_c.identifier
                    ]
                ] = np.zeros(dict_dummy_comp_r[flag_nodal]["temperature"].shape)

                # New solid-solid interface (cdp, 09/2020)
                htc_len = htc_len + 1
                if (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.CONDUCTION_HTC_COMPUTED
                ):

                    # Aliases
                    # Thermal contact resistance between s_comp_r and
                    # s_comp_c.
                    R_contact = conductor.coupling.thermal_contact_resistance[
                            s_comp_r.identifier,s_comp_c.identifier
                        ] # m^2K/W
                    # Thickness of component s_comp_r when in contact with
                    # component s_comp_c. It is assumed constant, even
                    # though it may change in the case of variable contact
                    # perimeter.
                    thick_s_comp_r_c = conductor.coupling.interface_thickness[
                            s_comp_r.identifier,s_comp_c.identifier
                        ] # m
                    # Thickness of component s_comp_c when in contact with
                    # component s_comp_r. It is assumed constant, even
                    # though it may change in the case of variable contact
                    # perimeter.
                    thick_s_comp_c_r = conductor.coupling.interface_thickness[
                            s_comp_c.identifier,s_comp_r.identifier
                        ] # m
                    # Thermal conductivity of s_comp_c
                    kk_s_comp_c = dict_dummy_comp_c[flag_nodal][
                        "total_thermal_conductivity"
                    ] # W/m/K

                    # Evaluate thermal resistance of c_comp_r.
                    R_s_comp_r = thick_s_comp_r_c/kk_s_comp_r # m^2K/W
                    # Evaluate thermal resistance of c_comp_c.
                    R_s_comp_c = thick_s_comp_c_r/kk_s_comp_c # m^2K/W

                    # Evaluate variable conductive heat transfer
                    # coefficient W/m^2/K.
                    htc_solid = 1.0 / (R_s_comp_r + R_contact + R_s_comp_c)

                    # Assign variable conductive heat transfer coefficient.
                    # Assumptions:
                    #   1) constant interface thickness, it may be actually
                    # variable in case of variable contact perimeter;
                    #   2) constant multiplier, it may be actually a
                    # function of the contact perimeter, temperature and
                    # magnetic field.
                    dict_dummy.HTC["sol_sol"]["cond"][
                        conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                            s_comp_c.identifier
                        ]
                    ] = mlt * htc_solid
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.CONDUCTION_HTC_READ_FROM_FILE
                ):
                    # Thermal contact.
                    dict_dummy.HTC["sol_sol"]["cond"][
                        conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                            s_comp_c.identifier
                        ]
                    ] = conductor.coupling.htc_contact[
                        s_comp_r.identifier, s_comp_c.identifier
                    ] * np.ones(
                        dict_dummy_comp_r[flag_nodal]["temperature"].shape
                    ) * mlt
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.RADIATIVE_HTC_COMPUTED
                ):
                    # Radiative heat transfer.
                    if (
                        s_comp_r.inputs.emissivity > 0.0
                        and s_comp_c.inputs.emissivity > 0.0
                        and conductor.coupling.view_factors[
                            s_comp_r.identifier, s_comp_c.identifier
                        ]
                        > 0.0
                    ):
                        # Evaluate the radiative heat transfer coefficient (assume that sr is the inner convex surface and sc the outer not convex surface, only in the comment below):
                        # A_sr*sigma*(T_sr^2 + T_sc^2)*(T_sr + T_sc)/((1 - emissivity_sr)/emissivity_sr + 1/F_sr_cs + (1 - emissivity_sc)/emissivity_sc*(A_sr/A_sc))
                        # Reciprocal of the view factor.
                        view_factor_rec = np.reciprocal(
                            conductor.coupling.view_factors[
                                s_comp_r.identifier, s_comp_c.identifier
                            ]
                        )
                        if (
                            s_comp_r.inputs.outer_perimeter
                            < s_comp_c.inputs.inner_perimeter
                        ):
                            # Set the contact perimeter to the correct value (overwrite the value assigned in input file conductor_coupling.xlsx)
                            conductor.coupling.contact_perimeter[
                                s_comp_r.identifier, s_comp_c.identifier
                            ] = s_comp_r.inputs.outer_perimeter
                            dict_dummy.HTC["sol_sol"]["rad"][
                                conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                                    s_comp_c.identifier
                                ]
                            ] = thermal_radiation.inner_radiative_htc(
                                s_comp_r,
                                s_comp_c,
                                dict_dummy_comp_r[flag_nodal]["temperature"],
                                dict_dummy_comp_c[flag_nodal]["temperature"],
                                view_factor_rec,
                            )
                        elif (
                            s_comp_c.inputs.outer_perimeter
                            < s_comp_r.inputs.inner_perimeter
                        ):
                            # Set the contact perimeter to the correct value (overwrite the value assigned in input file conductor_coupling.xlsx)
                            conductor.coupling.contact_perimeter[
                                s_comp_r.identifier, s_comp_c.identifier
                            ] = s_comp_c.inputs.outer_perimeter
                            dict_dummy.HTC["sol_sol"]["rad"][
                                conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                                    s_comp_c.identifier
                                ]
                            ] = thermal_radiation.inner_radiative_htc(
                                s_comp_c,
                                s_comp_r,
                                dict_dummy_comp_c[flag_nodal]["temperature"],
                                dict_dummy_comp_r[flag_nodal]["temperature"],
                                view_factor_rec,
                            ) * mlt
                        # End if s_comp_r.inputs.outer_perimeter.
                    # End if emissivity.
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                    )
                    == HTC_Choice.RADIATIVE_HTC_READ_FROM_FILE
                ):
                    # Radiative heat transfer from sheet contact_HTC of file conductor_coupling.xlsx.
                    dict_dummy.HTC["sol_sol"]["rad"][
                        conductor.dict_topology["sol_sol"][s_comp_r.identifier][
                            s_comp_c.identifier
                        ]
                    ] = conductor.coupling.htc_contact[
                        s_comp_r.identifier, s_comp_c.identifier
                    ] * np.ones(
                        dict_dummy_comp_r[flag_nodal]["temperature"].shape
                    ) * mlt
                # End if conductor.coupling.htc_choice[s_comp_r.identifier, s_comp_c.identifier]
        # end for loop cc
    # end for loop rr
    return htc_len


def evaluate_environment_solid_htc(conductor: object, simulation: object, dict_dummy: object, flag_nodal: bool) -> int:
    """Evaluate the HTC (``dict_dummy.HTC["env_sol"]``) at every environment-solid interface.

    Supports convective, radiative, and mixed convective-and-radiative heat
    transfer with the environment, each either code-computed or read from
    file. Only jackets of kind ``"outer_insulation"`` or ``"whole_enclosure"``
    may exchange heat with the environment.

    Returns:
        int: number of environment-solid interfaces evaluated.
    """
    interf_flag = conductor.coupling.contact_perimeter_flag
    htc_len = 0

    for s_comp_r in conductor.inventory.solids.collection:
        dict_dummy_comp_r = {
            True: s_comp_r.node_fields,
            False: s_comp_r.gauss_fields,
        }

        key = f"{simulation.environment.KIND}_{s_comp_r.identifier}"
        # Multiplier used in both cases (positive and negative flag).
        mlt = conductor.coupling.htc_multiplier[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
        if (
            abs(interf_flag[
                    simulation.environment.KIND, s_comp_r.identifier
                ]
            ) == 1
        ):
            # New environment-solid interface
            htc_len = htc_len + 1
            if (
                s_comp_r.inputs.jacket_kind == "outer_insulation"
                or s_comp_r.inputs.jacket_kind == "whole_enclosure"
            ):
                # Heat transfer with the environment by radiation and/or by convection.
                # Initialize dictionary.
                dict_dummy.HTC["env_sol"][key] = dict(
                    conv=np.zeros(
                        dict_dummy_comp_r[flag_nodal]["temperature"].size
                    ),
                    rad=np.zeros(dict_dummy_comp_r[flag_nodal]["temperature"].size),
                )
                if (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_COMPUTED
                ):
                    # Heat transfer by convection: heat transfer coefficient evaluated from air properties.
                    if conductor.inputs.is_rectangular:
                        # Rectangular conductor
                        dict_dummy.HTC["env_sol"][key]["conv"] = dict(
                            side=np.zeros(
                                dict_dummy_comp_r[flag_nodal]["temperature"].size
                            ),
                            bottom=np.zeros(
                                dict_dummy_comp_r[flag_nodal]["temperature"].size
                            ),
                            top=np.zeros(
                                dict_dummy_comp_r[flag_nodal]["temperature"].size
                            ),
                        )
                        # Evaluate side bottom and top surfaces htc.
                        (
                            dict_dummy.HTC["env_sol"][key]["conv"]["side"],
                            dict_dummy.HTC["env_sol"][key]["conv"]["bottom"],
                            dict_dummy.HTC["env_sol"][key]["conv"]["top"],
                        ) = simulation.environment.eval_heat_transfer_coefficient(
                            conductor, dict_dummy_comp_r[flag_nodal]["temperature"]
                        ) * mlt
                    else:
                        # Circular conductor
                        if flag_nodal == False:
                            # Compute only in gauss node to avoid error
                            dict_dummy.HTC["env_sol"][key]["conv"] = (
                                simulation.environment.eval_heat_transfer_coefficient(
                                    conductor,
                                    dict_dummy_comp_r[flag_nodal]["temperature"],
                                )
                                * conductor.inputs.phi_convective
                            )
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.CONVECTION_HTC_READ_FROM_FILE
                ):
                    # Heat transfer by convection: from sheet contact_HTC of file conductor_coupling.xlsx.
                    dict_dummy.HTC["env_sol"][key]["conv"] = (
                        conductor.coupling.htc_contact[
                            simulation.environment.KIND, s_comp_r.identifier
                        ]
                        * conductor.inputs.phi_convective
                        * np.ones(
                            dict_dummy_comp_r[flag_nodal]["temperature"].shape
                        ) * mlt
                    )
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.RADIATIVE_HTC_COMPUTED
                ):
                    # Heat transfer by radiation.
                    # Evaluate radiative heat transfer coefficient invoking method eval_weighted_radiative_htc.
                    dict_dummy.HTC["env_sol"][key][
                        "rad"
                    ] = thermal_radiation.eval_weighted_radiative_htc(
                        conductor,
                        simulation,
                        s_comp_r,
                        dict_dummy_comp_r[flag_nodal]["temperature"],
                    ) * mlt
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.RADIATIVE_HTC_READ_FROM_FILE
                ):
                    # Heat transfer by radiation: from sheet contact_HTC of file conductor_coupling.xlsx.
                    dict_dummy.HTC["env_sol"][key]["rad"] = (
                        conductor.coupling.htc_contact[
                            simulation.environment.KIND, s_comp_r.identifier
                        ]
                        * conductor.inputs.phi_radiative
                        * np.ones(
                            dict_dummy_comp_r[flag_nodal]["temperature"].shape
                        ) * mlt
                    )
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.MIXED_CONVECTION_AND_RADIATION_COMPUTED
                ):
                    # Heat transfer by radiation and free convection: code evaluation
                    dict_dummy.HTC["env_sol"][key]["conv"] = (
                        simulation.environment.eval_heat_transfer_coefficient(
                            conductor, dict_dummy_comp_r[flag_nodal]["temperature"]
                        )
                        * conductor.inputs.phi_convective * mlt
                    )

                    dict_dummy.HTC["env_sol"][key][
                        "rad"
                    ] = thermal_radiation.eval_weighted_radiative_htc(
                        conductor,
                        simulation,
                        s_comp_r,
                        dict_dummy_comp_r[flag_nodal]["temperature"],
                    ) * mlt
                elif (
                    HTC_Choice.get_htc_choice_flag(
                        conductor.coupling.htc_choice[
                        simulation.environment.KIND, s_comp_r.identifier
                    ]
                    )
                    == HTC_Choice.MIXED_CONVECTION_AND_RADIATION_FROM_FILE
                ):
                    # Heat transfer by radiation and free convection: from sheet contact_HTC of file conductor_coupling.xlsx.
                    # Questo va ragionato meglio: secondo me devo trovare il modo di distinguere i due contributi anche se in input sono dati come valore complessivo.
                    dict_dummy.HTC["env_sol"][key][
                        "conv"
                    ] = conductor.coupling.htc_contact[
                        simulation.environment.KIND, s_comp_r.identifier
                    ] * np.ones(
                        dict_dummy_comp_r[flag_nodal]["temperature"].shape
                    ) * mlt
                # End if conductor.coupling.htc_choice[simulation.environment.KIND, s_comp_r.identifier]
            else:
                # Raise error
                raise os.error(
                    f"JacketComponent of kind {s_comp_r.inputs.jacket_kind} can not exchange heat by radiation and/or convection with the environment.\n"
                )
            # End if s_comp_r.inputs.jacket_kind
        # End if abs(intef_flag.at[simulation.environment.KIND, s_comp_r.identifier])
    # end for loop s_comp_r.
    return htc_len
