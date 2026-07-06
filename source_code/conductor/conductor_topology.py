"""
This module owns component-pair topology discovery for a conductor: which
channels are in hydraulic parallel or only in thermal contact, which channels
are standalone, and the full interface registry between fluid, solid, and
environment components (contact perimeters and the ``interface`` /
``dict_topology`` data structures).

Relocated from ``conductor/conductor.py`` as part of splitting the file for
single responsibility. This is a cohesive, mutually-recursive graph-discovery
subsystem (``search_on_ind_col``/``search_on_ind_row`` call each other), kept
together in this one module.
"""

from collections import namedtuple
import os

import numpy as np
from scipy import interpolate

from conductor.conductor_flags import ContactPerimeterFlag


def conductors_coupling(conductor: object) -> None:
    """Placeholder for coupling between different Conductor objects (not yet implemented)."""
    pass


def get_conductor_topology(conductor: object, environment: object) -> None:

    """
    Function that evaluate the detailed conductor topology (dict_topology) together with the contatc perimeter (dict_interf_peri). Possible configurations are:
    1) channels in hydraulic parallel;
    2) channels not in hydraulic parallel but in thermal contact;
    3) stand alone channels (no mass or energy exchange);
    4) channels in thermal contact with solid components;
    5) channels not in thermal contact with solid components;
    6) contact between solid components.

    dict_topology describes the full conductor topology, i.e. interface between channels, channels and solid components, solid components as well as the isolated channels; it is organized into three sub dictionaries accessed by keys "ch_ch"; "ch_sol"; "sol_sol".
    Each sub dictionary is characterized by a series of strings that uniquely determines which components are in contact, and a list of object constituted by all the components in contact. Keys of this dictionaries are the identifier of the first object in alphabetical order constituting the interface.

    dict_interf_peri holds the values of the contact perimenter. It is also subdivided into three sub dictionaries with the same name as above. It is important to notice that in case of contact between channels, the "Open" and "Close" keys are introduced.
    (cdp, 09/2020)
    """

    # Alias
    interf_flag = conductor.coupling.contact_perimeter_flag

    # nested dictionaries declarations
    conductor.dict_topology["ch_ch"] = dict()
    conductor.dict_topology["ch_ch"]["Hydraulic_parallel"] = dict()
    conductor.dict_topology["ch_ch"]["Thermal_contact"] = dict()
    conductor.dict_topology["Standalone_channels"] = list()
    conductor.dict_interf_peri["ch_ch"] = dict()
    conductor.dict_interf_peri["ch_ch"]["Open"] = dict(
        nodal=dict(),
        Gauss=dict(),
    )
    conductor.dict_interf_peri["ch_ch"]["Close"] = dict(
        nodal=dict(),
        Gauss=dict(),
    )
    conductor.dict_topology["ch_sol"] = dict()
    conductor.dict_interf_peri["ch_sol"] = dict(
        nodal=dict(),
        Gauss=dict(),
    )
    conductor.dict_topology["sol_sol"] = dict()
    conductor.dict_interf_peri["sol_sol"] = dict(
        nodal=dict(),
        Gauss=dict(),
    )
    conductor.dict_interf_peri["env_sol"] = dict(
        nodal=dict(),
        Gauss=dict(),
    )

    # Call function get_hydraulic_parallel to obtain the channels subdivision \
    # into groups of channels that are in hydraulic parallel.
    get_hydraulic_parallel(conductor)

    # Nested loop channel-channel (cdp, 09/2020)
    for rr, fluid_comp_r in enumerate(conductor.inventory.fluids.collection):
        for cc, fluid_comp_c in enumerate(
            conductor.inventory.fluids.collection[rr + 1 :], rr + 1
        ):
            if (
                abs(interf_flag[
                    fluid_comp_r.identifier, fluid_comp_c.identifier
                ]
                ) == 1
            ):
                # There is at least thermal contact between fluid_comp_r
                # and fluid_comp_c
                # Assign the contact perimeter value
                (
                    conductor.dict_interf_peri["ch_ch"]["Close"],
                    conductor.dict_interf_peri["ch_ch"]["Open"]
                ) = _assign_contact_perimeter_fluid_comps(
                    conductor,
                    fluid_comp_r.identifier,
                    fluid_comp_c.identifier,
                )
                if cc == rr + 1:
                    # declare dictionary flag_found (cdp, 09/2020)
                    flag_found = dict()
                # Invoke function get_thermal_contact_channels to search for channels \
                # that are only in thermal contact (cdp, 09/2020)
                flag_found = get_thermal_contact_channels(
                    conductor, rr, cc, fluid_comp_r, fluid_comp_c, flag_found
                )
            # end abs(interf_flag[fluid_comp_r.identifier, fluid_comp_c.identifier]) == 1 (cdp, 09/2020)
        # end for cc (cdp, 09/2020)
        if (
            conductor.dict_topology["ch_ch"]["Thermal_contact"].get(
                fluid_comp_r.identifier
            )
            != None
        ):
            # key fluid_comp_r.identifier exists (cdp, 09/2020)
            if (
                len(
                    list(
                        conductor.dict_topology["ch_ch"]["Thermal_contact"][
                            fluid_comp_r.identifier
                        ].keys()
                    )
                )
                - 3
                > 0
            ):
                # There are channels that are in thermal contact (cdp, 09/2020)
                # Update the number of channels in thermal contact with fluid_comp_r: it \
                # is the length of list group. The actual number of thermal contacts \
                # can be larger, since some channels that are in thermal contact may \
                # belong to groups of channels in hydraulic parallel; it is keep \
                # into account by the key Actual_number. (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"][
                    fluid_comp_r.identifier
                ].update(
                    Number=len(
                        conductor.dict_topology["ch_ch"]["Thermal_contact"][
                            fluid_comp_r.identifier
                        ]["Group"]
                    )
                )
                # Assign values to key Actual_number (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"][
                    fluid_comp_r.identifier
                ].update(
                    Actual_number=len(
                        list(
                            conductor.dict_topology["ch_ch"]["Thermal_contact"][
                                fluid_comp_r.identifier
                            ].keys()
                        )
                    )
                    - 3
                    + 1
                )
            else:
                # There are not channels that are in thermal contact remove \
                # key fluid_comp_r.identifier from dictionary \
                # conductor.dict_topology["ch_ch"]["Thermal_contact"] (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"].pop(
                    fluid_comp_r.identifier
                )
        # end if conductor.dict_topology["ch_ch"]\
        # ["Thermal_contact"].get(fluid_comp_r.identifier) != None (cdp, 09/2020)
    # end for rr (cdp, 09/2020)
    # Call function find_standalone_channels to search for eventually stand \
    # alone channels (not in hydraulic parallel) (cdp, 09/2020)
    find_standalone_channels(conductor)
    # dummy dictionary to store channel-solid topology (cdp, 09/2020)
    dict_topology_dummy_ch_sol = dict()
    # dummy to optimize nested loop (cdp, 09/2020)
    dict_chan_s_comp_contact = dict()
    # Nested loop channel-solid (cdp, 09/2020)
    for _, fluid_comp_r in enumerate(conductor.inventory.fluids.collection):
        # List linked channels-solid initialization (cdp, 09/2020)
        list_linked_chan_sol = list()
        # Nested dictionary in dict_topology_dummy_ch_sol declaration \
        # dict_topology_dummy_ch_sol
        dict_topology_dummy_ch_sol[fluid_comp_r.identifier] = dict()
        for _, s_comp_c in enumerate(conductor.inventory.solids.collection):
            if (
                abs(interf_flag[
                        fluid_comp_r.identifier, s_comp_c.identifier
                    ]
                ) == 1
            ):
                # There is contact between fluid_comp_r and s_comp_c

                conductor.dict_interf_peri["ch_sol"] = _assign_contact_perimeter_not_fluid_only(
                    conductor,
                    fluid_comp_r.identifier,
                    s_comp_c.identifier,
                    "ch_sol",
                )

                # Interface identification (cdp, 09/2020)
                dict_topology_dummy_ch_sol[fluid_comp_r.identifier][
                    s_comp_c.identifier
                ] = f"{fluid_comp_r.identifier}_{s_comp_c.identifier}"
                # Call function chan_sol_interfaces (cdp, 09/2020)
                [
                    dict_chan_s_comp_contact,
                    list_linked_chan_sol,
                ] = chan_sol_interfaces(
                    fluid_comp_r,
                    s_comp_c,
                    dict_chan_s_comp_contact,
                    list_linked_chan_sol,
                )
            # end if abs(interf_flag[fluid_comp_r.identifier, s_comp_c.identifier]) == 1: (cdp, 09/2020)
        # end for cc (cdp, 09/2020)
        # Call function update_interface_dictionary to update dictionaries \
        # (cdp, 09/2020)
        [
            dict_topology_dummy_ch_sol,
            dict_chan_s_comp_contact,
        ] = update_interface_dictionary(
            fluid_comp_r,
            dict_topology_dummy_ch_sol,
            dict_chan_s_comp_contact,
            list_linked_chan_sol,
        )
    # end for rr (cdp, 09/2020)
    conductor.dict_topology.update(ch_sol=dict_topology_dummy_ch_sol)
    # dummy dictionary to store solid-solid topology (cdp, 09/2020)
    dict_topology_dummy_sol = dict()
    # dummy to optimize nested loop (cdp, 09/2020)
    dict_s_comps_contact = dict()
    # Nested loop solid-solid (cdp, 09/2020)
    for rr, s_comp_r in enumerate(conductor.inventory.solids.collection):
        # List linked solids initialization (cdp, 09/2020)
        list_linked_solids = list()
        # Nested dictionary in dict_topology_dummy_sol declaration \
        # dict_topology_dummy_sol
        dict_topology_dummy_sol[s_comp_r.identifier] = dict()
        for _, s_comp_c in enumerate(
            conductor.inventory.solids.collection[rr + 1 :]
        ):
            if (
                abs(interf_flag[
                        s_comp_r.identifier, s_comp_c.identifier
                    ]
                ) == 1
            ):
                # There is contact between s_comp_r and s_comp_c

                conductor.dict_interf_peri["sol_sol"] = _assign_contact_perimeter_not_fluid_only(
                    conductor,
                    s_comp_r.identifier,
                    s_comp_c.identifier,
                    "sol_sol",
                )

                conductor.dict_interf_peri["sol_sol"][
                    f"{s_comp_r.identifier}_{s_comp_c.identifier}"
                ] = conductor.coupling.contact_perimeter[
                    s_comp_r.identifier, s_comp_c.identifier
                ]
                # Interface identification (cdp, 09/2020)
                dict_topology_dummy_sol[s_comp_r.identifier][
                    s_comp_c.identifier
                ] = f"{s_comp_r.identifier}_{s_comp_c.identifier}"
                # Call function chan_sol_interfaces (cdp, 09/2020)
                [
                    dict_s_comps_contact,
                    list_linked_solids,
                ] = chan_sol_interfaces(
                    s_comp_r, s_comp_c, dict_s_comps_contact, list_linked_solids
                )
            # end if abs(interf_flag[rr, cc]) == 1: (cdp, 09/2020)
        # end for cc (cdp, 09/2020)
        # Call function update_interface_dictionary to update dictionaries \
        # (cdp, 09/2020)
        [
            dict_topology_dummy_sol,
            dict_s_comps_contact,
        ] = update_interface_dictionary(
            s_comp_r,
            dict_topology_dummy_sol,
            dict_s_comps_contact,
            list_linked_solids,
        )
        if (
            abs(interf_flag[environment.KIND,s_comp_r.identifier]) == 1
        ):
            if (
                s_comp_r.inputs.jacket_kind == "outer_insulation"
                or s_comp_r.inputs.jacket_kind == "whole_enclosure"
            ):
                # There is an interface between environment and s_comp_r.
                conductor.dict_interf_peri["env_sol"] = _assign_contact_perimeter_not_fluid_only(
                    conductor,
                    environment.KIND,
                    s_comp_r.identifier,
                    "env_sol",
                )
            else:
                # Raise error
                raise os.error(
                    f"JacketComponent of kind {s_comp_r.inputs.jacket_kind} can not have and interface with the environment.\n"
                )
            # End if s_comp_r.inputs.jacket_kind
    # end for rr (cdp, 09/2020)
    conductor.dict_topology.update(sol_sol=dict_topology_dummy_sol)


def _create_constant_perimeter(conductor: object, comp1_id: str, comp2_id: str):
    contact_perimeter = conductor.coupling.contact_perimeter[
        comp1_id, comp2_id
    ]
    return (
        np.full(conductor.mesh.number_of_nodes, contact_perimeter, dtype=float),
        np.full(conductor.mesh.number_of_elements, contact_perimeter, dtype=float),
    )


def _interpolate_contact_perimeter(
    conductor: object, comp1_id: str, comp2_id: str, values: np.ndarray
) -> dict:
    points = conductor.dict_df_variable_contact_perimeter[comp1_id].iloc[:, 0].to_numpy(
        dtype=float
    )
    interpolator = interpolate.interp1d(
        points,
        values,
        bounds_error=False,
        fill_value=values[-1],
        kind="linear",
    )
    return {
        "nodal": interpolator(conductor.mesh.node_coordinates),
        "Gauss": interpolator(conductor.mesh.gauss_point_coordinates),
    }


def _assign_contact_perimeter_fluid_comps(
    conductor: object,
    comp1_id: str,
    comp2_id: str,
) -> tuple:
    interf_peri_open = conductor.dict_interf_peri["ch_ch"]["Open"]
    interf_peri_close = conductor.dict_interf_peri["ch_ch"]["Close"]
    interf_flag = conductor.coupling.contact_perimeter_flag[
        comp1_id, comp2_id
    ]
    open_fraction = conductor.coupling.open_perimeter_fraction[
        comp1_id, comp2_id
    ]

    if interf_flag == ContactPerimeterFlag.CONSTANT_CONTACT_PERIMETER:
        nodal, gauss = _create_constant_perimeter(conductor, comp1_id, comp2_id)
        interf_peri_open["nodal"][f"{comp1_id}_{comp2_id}"] = (
            nodal * open_fraction
        )
        interf_peri_open["Gauss"][f"{comp1_id}_{comp2_id}"] = (
            gauss * open_fraction
        )
        interf_peri_close["nodal"][f"{comp1_id}_{comp2_id}"] = (
            nodal * (1.0 - open_fraction)
        )
        interf_peri_close["Gauss"][f"{comp1_id}_{comp2_id}"] = (
            gauss * (1.0 - open_fraction)
        )
    elif interf_flag == ContactPerimeterFlag.VARIABLE_CONTACT_PERIMETER:
        raw = conductor.dict_df_variable_contact_perimeter[comp1_id].loc[
            :, comp2_id
        ].to_numpy(dtype=float)
        open_values = raw * open_fraction
        close_values = raw * (1.0 - open_fraction)
        open_interp = _interpolate_contact_perimeter(
            conductor, comp1_id, comp2_id, open_values
        )
        close_interp = _interpolate_contact_perimeter(
            conductor, comp1_id, comp2_id, close_values
        )
        interf_peri_open["nodal"][f"{comp1_id}_{comp2_id}"] = open_interp["nodal"]
        interf_peri_open["Gauss"][f"{comp1_id}_{comp2_id}"] = open_interp["Gauss"]
        interf_peri_close["nodal"][f"{comp1_id}_{comp2_id}"] = close_interp["nodal"]
        interf_peri_close["Gauss"][f"{comp1_id}_{comp2_id}"] = close_interp["Gauss"]

    return interf_peri_close, interf_peri_open


def _assign_contact_perimeter_not_fluid_only(
    conductor: object,
    comp1_id: str,
    comp2_id: str,
    interf_kind: str,
) -> dict:
    interf_peri = conductor.dict_interf_peri[interf_kind]
    interf_flag = conductor.coupling.contact_perimeter_flag[
        comp1_id, comp2_id
    ]

    if interf_flag == ContactPerimeterFlag.CONSTANT_CONTACT_PERIMETER:
        nodal, gauss = _create_constant_perimeter(conductor, comp1_id, comp2_id)
        interf_peri["nodal"][f"{comp1_id}_{comp2_id}"] = nodal
        interf_peri["Gauss"][f"{comp1_id}_{comp2_id}"] = gauss
    elif interf_flag == ContactPerimeterFlag.VARIABLE_CONTACT_PERIMETER:
        raw = conductor.dict_df_variable_contact_perimeter[comp1_id].loc[
            :, comp2_id
        ].to_numpy(dtype=float)
        interpolated = _interpolate_contact_perimeter(
            conductor, comp1_id, comp2_id, raw
        )
        interf_peri["nodal"][f"{comp1_id}_{comp2_id}"] = interpolated["nodal"]
        interf_peri["Gauss"][f"{comp1_id}_{comp2_id}"] = interpolated["Gauss"]

    return interf_peri


def chan_sol_interfaces(
    comp_r, comp_c, dict_comp_interface, list_linked_comp
):

    """
    Function that evaluates interfaces between channels and solid components or between solids, and list them in a list of objects to be assigned to dict_topology. (cdp, 09/2020)
    """

    if dict_comp_interface.get(comp_r.identifier) == None:
        # No key called comp_r.identifier in dictionary dict_comp_interface \
        # (cdp, 09/2020)
        dict_comp_interface[comp_r.identifier] = list()
        # In this case necessarily we store both comp_r and comp_c \
        # (cdp, 09/2020)
        list_linked_comp.append(comp_r)
        list_linked_comp.append(comp_c)
    else:  # key comp_r.identifier already exist in dict_comp_interface
        # In this case store necessarily only comp_c (cdp, 09/2020)
        list_linked_comp.append(comp_c)
    # end if dict_comp_interface.get(comp_r.identifier) (cdp, 09/2020)
    return [dict_comp_interface, list_linked_comp]

# end function chan_sol_interfaces (cdp, 09/2020)


def find_standalone_channels(conductor: object) -> None:

    """
    Function that searchs for possible isolated (not in hydraulic parallel) channels: search is on each channel in order to not miss anything (cdp, 09/2020)
    """

    # crate dictionary used to understand if channel is or not a stand alone one
    check_found = dict()

    ii = -1
    while ii < conductor.inventory.fluids.number - 1:
        ii = ii + 1
        fluid_comp = conductor.inventory.fluids.collection[ii]
        # loop on reference channels (cdp, 09/2020)
        check_found[fluid_comp.identifier] = dict(
            Hydraulic_parallel=False, Thermal_contact=False
        )
        for fluid_comp_ref in list(
            conductor.dict_topology["ch_ch"]["Hydraulic_parallel"].keys()
        ):
            # Search in Hydraulic parallel groups (cdp, 09/2020)
            if check_found[fluid_comp.identifier]["Hydraulic_parallel"] == False:
                if (
                    fluid_comp
                    in conductor.dict_topology["ch_ch"]["Hydraulic_parallel"][
                        fluid_comp_ref
                    ]["Group"]
                ):
                    # channel fluid_comp constitutes a group of channels in hydraulic \
                    # parallel thus it can not be a stand alone channel (cdp, 09/2020)
                    # Update dictionart check_found (cdp, 09/2020)
                    check_found[fluid_comp.identifier].update(
                        Hydraulic_parallel=True
                    )
        if check_found[fluid_comp.identifier]["Hydraulic_parallel"] == False:
            # Channel fluid_comp is not inside Hydraulic parallel groups (cdp, 09/2020)
            for fluid_comp_ref in list(
                conductor.dict_topology["ch_ch"]["Thermal_contact"].keys()
            ):
                # Search in Hydraulic parallel groups (cdp, 09/2020)
                if check_found[fluid_comp.identifier]["Thermal_contact"] == False:
                    if (
                        fluid_comp
                        in conductor.dict_topology["ch_ch"]["Thermal_contact"][
                            fluid_comp_ref
                        ]["Group"]
                    ):
                        # channel fluid_comp constitutes a thermal contact thus it can not be \
                        # a stand alone channel (cdp, 09/2020)
                        # Update dictionart check_found (cdp, 09/2020)
                        check_found[fluid_comp.identifier].update(
                            Thermal_contact=True
                        )
        if (
            check_found[fluid_comp.identifier]["Hydraulic_parallel"] == False
            and check_found[fluid_comp.identifier]["Thermal_contact"] == False
        ):
            # fluid_comp is a stand alone channel since it does not belong to a group of \
            # channels in hydraulic parallel and it does not constitute a thermal \
            # contact (cdp, 09/2020)
            conductor.dict_topology["Standalone_channels"].append(fluid_comp)

# 	N_channel_no_par = len(conductor.dict_topology["Standalone_channels"])
# 	if N_channel_no_par == 0:
# 		print("There are no isolated channels\n")
# 	elif N_channel_no_par > 0 and N_channel_no_par < \
# 			 conductor.inventory.fluids.number:
# 		if N_channel_no_par == 1:
# 			print(f"""There is {N_channel_no_par} channel that is not in hydraulic parallel: {conductor.dict_topology["Standalone_channels"][0].identifier}\n""")
# 		else:
# 			print(f"""There are {N_channel_no_par} channels that are not in hydraulic parallel: {conductor.dict_topology["Standalone_channels"][:].identifier}\n""")
# 	elif N_channel_no_par == \
# 			 conductor.inventory.fluids.number:
# 		print("All channels are isolated\n")
# 	else:
# 		print(f"Something does not work\n")
# end function find_standalone_channels (cdp, 09/2020)


def update_interface_dictionary(
    comp, dict_topology_dummy, dict_contacts, list_contacts
):

    dict_contacts[comp.identifier] = list_contacts
    dict_topology_dummy[comp.identifier].update(Group=list_contacts)
    dict_topology_dummy[comp.identifier].update(Number=len(list_contacts))
    if dict_topology_dummy[comp.identifier]["Number"] == 0:
        # Removed empty keys from dictionaries (cdp, 09/2020)
        dict_topology_dummy.pop(comp.identifier)
        dict_contacts.pop(comp.identifier)
    return [dict_topology_dummy, dict_contacts]

# end function update_interface_dictionary (cdp, 09/2020)


def get_hydraulic_parallel(conductor: object) -> None:

    """
    Function that interprets the information in table conductor.coupling.open_perimeter_fraction understanding if there are channels in hydraulic parallel and how the are organized into groups. The function returns a dictionary with:
    1) the identifier of the reference channel of each group
    2) a list of all the channels that belongs to a group
    3) for each group the IDs of all the linked channels organized into lists
    (cdp, 09/2020)
    """

    full_ind = dict()
    # get row and column index of non zero matrix elements (cdp, 09/2020)
    [full_ind["row"], full_ind["col"]] = np.nonzero(
        conductor.coupling.open_perimeter_fraction.matrix[1:, 1:]
    )
    # USEFUL QUANTITIES AND VARIABLES (cdp, 09/2020)
    # array of the not considered array (cdp, 09/2020)
    already = dict(no=np.unique(np.union1d(full_ind["row"], full_ind["col"])))
    # list of the already considered array (cdp, 09/2020)
    already["yes"] = -1 * np.ones(already["no"].shape, dtype=int)
    # index to used to update key "yes" (cdp, 09/2020)
    already["ii_yes"] = 0
    # Define dictionary dict_topology. This is different from \
    # conductor.dict_topology which is a class Conductor attribute. A the end of \
    # this function conductor.dict_topology will be updated with the values in \
    # dict_topology (cdp, 09/2020)
    dict_topology = dict()
    # Define dictionary check to be sure that for each channel both searchs \
    # are performed (cdp, 09/2020)
    check = dict()

    Total_connections = len(full_ind["row"])
    total_connections_counter = 0
    group_counter = 0
    # loop until all channel connections are realized (cdp, 09/2020)
    while total_connections_counter < Total_connections:
        # get the reference channel: it is the one characterized by the minimum \
        # index value in array already["no"] (cdp, 09/2020)
        fluid_comp_ref_row_ind = min(already["no"])
        fluid_comp_ref = conductor.inventory.fluids.collection[
            fluid_comp_ref_row_ind
        ]
        # update dictionary already (cdp, 09/2020)
        already["no"] = np.delete(already["no"], 0, 0)
        if fluid_comp_ref_row_ind not in already["yes"]:
            already["yes"][already["ii_yes"]] = fluid_comp_ref_row_ind
            already.update(ii_yes=already["ii_yes"] + 1)
        # Construct check dictionary (cdp, 09/2020)
        check[fluid_comp_ref.identifier] = dict()
        # Get minimum and maximum index of array full_ind["row"] that correspond \
        # to the reference channel (cdp, 09/2020)
        boundary = dict(
            fluid_comp_ref_lower=min(
                np.nonzero(full_ind["row"] == fluid_comp_ref_row_ind)[0]
            ),
            fluid_comp_ref_upper=max(
                np.nonzero(full_ind["row"] == fluid_comp_ref_row_ind)[0]
            ),
        )
        # get all the channels that are directly in contact with reference \
        # channel (cdp, 09/2020)
        ind_direct = full_ind["col"][
            boundary["fluid_comp_ref_lower"] : boundary["fluid_comp_ref_upper"] + 1
        ]
        # Update dictionary dict_topology
        dict_topology[fluid_comp_ref.identifier] = dict(
            Ref_channel=fluid_comp_ref.identifier, Group=list(), Number=0
        )
        dict_topology[fluid_comp_ref.identifier]["Group"].append(fluid_comp_ref)
        for ch_index in ind_direct:
            # get channel (cdp, 09/2020)
            fluid_comp = conductor.inventory.fluids.collection[ch_index]
            # Construct check dictionary (cdp, 09/2020)
            check[fluid_comp_ref.identifier][fluid_comp.identifier] = dict(
                row=False, col=False
            )
            # find the index in array already["no"] of the element that must be \
            # deleted (cdp, 09/2020)
            i_del = np.nonzero(already["no"] == ch_index)[0]
            # update total_connections_counter (cdp, 09/2020)
            total_connections_counter = total_connections_counter + 1
            # update dictionary already (cdp, 09/2020)
            already["no"] = np.delete(already["no"], i_del, 0)
            if ch_index not in already["yes"]:
                already["yes"][already["ii_yes"]] = ch_index
                already.update(ii_yes=already["ii_yes"] + 1)
            dict_topology[fluid_comp_ref.identifier][fluid_comp.identifier] = [
                f"{fluid_comp_ref.identifier}_{fluid_comp.identifier}"
            ]
            dict_topology[fluid_comp_ref.identifier]["Group"].append(fluid_comp)
        # end for ii (cdp, 09/2020)
        # for each channel that is in direct contact with the reference one, \
        # search if it is in contact with other channels, constituting an \
        # indirect contact with the reference channel. This is done in a \
        # different loop because total_connections_counter must be fully updated \
        # (cdp, 09/2020)
        for ch_index in ind_direct:
            # get channel (cdp, 09/2020)
            fluid_comp = conductor.inventory.fluids.collection[ch_index]
            # Initialize key value "variable_lower" of dictionary boundary. This \
            # parameter is used to look only in the region of not directly \
            # connected channels and is updated to consider only the data below \
            # this index value. Initialization must be done at each iteration in \
            # order to not miss some index during the search. (cdp, 09/2020)
            boundary.update(variable_lower=boundary["fluid_comp_ref_upper"] + 1)
            if (
                check[fluid_comp_ref.identifier][fluid_comp.identifier]["col"]
                == False
            ):
                # The search in array full_ind["col"] is not performed yet \
                # (cdp, 09/2020)
                total_connections_counter = search_on_ind_col(
                    conductor,
                    ch_index,
                    full_ind,
                    dict_topology,
                    fluid_comp_ref,
                    fluid_comp,
                    check,
                    already,
                    total_connections_counter,
                    boundary,
                )
            if (
                check[fluid_comp_ref.identifier][fluid_comp.identifier]["row"]
                == False
            ):
                # The search in array full_ind["row"] is not performed yet \
                # (cdp, 09/2020)
                total_connections_counter = search_on_ind_row(
                    conductor,
                    ch_index,
                    full_ind,
                    dict_topology,
                    fluid_comp_ref,
                    fluid_comp,
                    check,
                    already,
                    total_connections_counter,
                    boundary,
                )
        # end for (cdp, 09/2020)
        # Sort list Group by channel identifier (cdp, 09/2020)
        dict_topology[fluid_comp_ref.identifier].update(
            Group=sorted(
                dict_topology[fluid_comp_ref.identifier]["Group"],
                key=lambda ch: ch.identifier,
            )
        )
        # Get the number of channels that are in hydraulic parallel for each \
        # reference channel (cdp, 10/2020)
        dict_topology[fluid_comp_ref.identifier].update(
            Number=len(dict_topology[fluid_comp_ref.identifier]["Group"])
        )
        if total_connections_counter == Total_connections:
            if group_counter == 0:
                # Only one group of channels in hydraulic parallel (cdp, 09/2020)
                group_counter = group_counter + 1
                print(
                    f"There is only {group_counter} group of channels in hydraulic parallel.\n"
                )
            elif group_counter > 0:
                # There are a least two groups of channels in hydraulic parallel \
                # (cdp, 09/2020)
                group_counter = group_counter + 1
                print(
                    f"There are {group_counter} groups of channels in hydraulic parallel\n"
                )
        elif total_connections_counter < Total_connections:
            # The number of groups of channels in hydraulic parallel is > 1.
            # Repeat the above procedure. Keep in mind that the reference channel \
            # is evaluated as the one with the minimum identifier between the ones that \
            # are not connected yet (cdp, 09/2020)
            group_counter = group_counter + 1
        elif total_connections_counter > Total_connections:
            # Something odd occurs
            raise ValueError(
                "ERROR! The counter of the total connections can not be larger than the number of total connections! Something odd occurs!\n"
            )
        # end if total_connections_counter (cdp, 09/2020)
    # end while (cdp, 09/2020)
    # Update key Hydraulic_parallel of dictionary dict_topology (cdp, 09/2020)
    conductor.dict_topology["ch_ch"].update(Hydraulic_parallel=dict_topology)

# end function get_hydraulic_parallel (cdp, 09/2020)


def search_on_ind_col(
    conductor,
    ch_index,
    full_ind,
    dict_topology,
    fluid_comp_ref,
    fluid_comp_c,
    check,
    already,
    total_connections_counter,
    boundary,
):

    """
    Function that search in array full_ind["col"] if there are other recall to channel fluid_comp_c (cdp, 09/2020)
    """

    # N.B ch_index is the channel that is in contact with the fluid_comp_ref or \
    # another channel (cdp, 09/2020)

    # The search in array full_ind["col"] will be performed so flag check \
    # [fluid_comp_ref.identifier][fluid_comp.identifier]["col"] can be set to True. (cdp, 09/2020)
    check[fluid_comp_ref.identifier][fluid_comp_c.identifier].update(col=True)
    # search for all the values that are equal to ch_index in array \
    # full_ind["col"], excluding the index of the direct contact region, and
    # store the corresponding indices (cdp, 09/2020)
    ind_found = (
        np.nonzero(full_ind["col"][boundary["variable_lower"] :] == ch_index)[0]
        + boundary["variable_lower"]
    )
    if len(ind_found) == 0:
        if (
            check[fluid_comp_ref.identifier][fluid_comp_c.identifier]["row"]
            == False
        ):
            # Search if there is some value equal to ch_index in the array \
            # full_ind["row"] calling function search_on_ind_row (cdp, 09/2020)
            total_connections_counter = search_on_ind_row(
                conductor,
                ch_index,
                full_ind,
                dict_topology,
                fluid_comp_ref,
                fluid_comp_c,
                check,
                already,
                total_connections_counter,
                boundary,
            )
        elif (
            check[fluid_comp_ref.identifier][fluid_comp_c.identifier]["row"] == True
        ):
            # no channel with smaller channel index is connected with ch_index \
            # (cdp, 09/2020)
            print(f"No other channels are connected to {fluid_comp_c.identifier}\n")
    elif len(ind_found) > 0:
        # there is at least one channel with smaller channel index that is \
        # connected with ch_index (cdp, 09/2020)
        # update key "variable_lower": the next search is done from this index \
        # value up to the end of arrays full_ind["row"] and full_ind["col"] \
        # (cdp, 09/2020)
        boundary.update(variable_lower=ind_found[0] + 1)
        # get the index of the linked channel(s) (col -> row) (cdp, 09/2020)
        ind_link = full_ind["row"][ind_found[0 : len(ind_found)]]
        # loop linked channels (cdp, 09/2020)
        jj = -1
        while jj < len(ind_link) - 1:
            jj = jj + 1
            ch_index = ind_link[jj]
            # get channel
            fluid_comp_r = conductor.inventory.fluids.collection[ch_index]
            if (
                dict_topology[fluid_comp_ref.identifier].get(
                    fluid_comp_r.identifier
                )
                != None
                and f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                in dict_topology[fluid_comp_ref.identifier][fluid_comp_r.identifier]
            ):
                # Skip operations since the connection is already preformed \
                # (cdp, 09/2020)
                jj = jj + 1
            else:
                if (
                    check[fluid_comp_ref.identifier].get(fluid_comp_r.identifier)
                    == None
                ):
                    # Update dictionary check: add key fluid_comp_r.identifier (cdp, 09/2020)
                    check[fluid_comp_ref.identifier][
                        fluid_comp_r.identifier
                    ] = dict(row=False, col=False)
                # construct channels link (cdp, 09/2020)
                if (
                    dict_topology[fluid_comp_ref.identifier].get(
                        fluid_comp_r.identifier
                    )
                    == None
                ):
                    # key fluid_comp_r.identifier does not exist, so it is added to the dictionary \
                    # and a list is created (cdp, 09/2020)
                    dict_topology[fluid_comp_ref.identifier][
                        fluid_comp_r.identifier
                    ] = [f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"]
                    print(
                        dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ][-1]
                        + "\n"
                    )
                    # update total_connections_counter (cdp, 09/2020)
                    total_connections_counter = total_connections_counter + 1
                else:
                    # key fluid_comp_r.identifier exists (cdp, 09/2020)
                    if (
                        f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                        not in dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ]
                    ):
                        # List is updated only if the contact identifier is not already in the \
                        # list (cdp, 09/2020)
                        dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ].append(
                            f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                        )
                        print(
                            dict_topology[fluid_comp_ref.identifier][
                                fluid_comp_r.identifier
                            ][-1]
                            + "\n"
                        )
                        # update total_connections_counter (cdp, 09/2020)
                        total_connections_counter = total_connections_counter + 1
                # end if dict_topology[fluid_comp_ref.identifier].get(fluid_comp_r.identifier) == None \
                # (cdp, 09/2020)
                # find the index in array already["no"] of the element that must be \
                # deleted (cdp, 09/2020)
                i_del = np.nonzero(already["no"] == ch_index)[0]
                # update dictionary already (cdp, 09/2020)
                already["no"] = np.delete(already["no"], i_del, 0)
                if ch_index not in already["yes"]:
                    already["yes"][already["ii_yes"]] = ch_index
                    already.update(ii_yes=already["ii_yes"] + 1)
                if (
                    fluid_comp_r
                    not in dict_topology[fluid_comp_ref.identifier]["Group"]
                ):
                    # Add channel fluid_comp_r to list Group (cdp, 09/2020)
                    dict_topology[fluid_comp_ref.identifier]["Group"].append(
                        fluid_comp_r
                    )
                # call function search_on_ind_row with to search if channel fluid_comp_r \
                # is linked to other channels (cdp, 09/2020)
                total_connections_counter = search_on_ind_row(
                    conductor,
                    ch_index,
                    full_ind,
                    dict_topology,
                    fluid_comp_ref,
                    fluid_comp_r,
                    check,
                    already,
                    total_connections_counter,
                    boundary,
                )
            # end if dict_topology[fluid_comp_ref.identifier].get(fluid_comp_r.identifier) != None and \
            # f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}" in dict_topology[fluid_comp_ref.identifier][fluid_comp_r.identifier] \
            # (cdp, 09/2020)
        # end while (cdp, 09/2020)
    # end if len(ind_found) (cdp, 09/2020)
    return total_connections_counter

# end function search_on_ind_col (cdp, 09/2020)


def search_on_ind_row(
    conductor,
    ch_index,
    full_ind,
    dict_topology,
    fluid_comp_ref,
    fluid_comp_r,
    check,
    already,
    total_connections_counter,
    boundary,
):

    """
    Function that search in array full_ind["row"] if there are other recall to channel fluid_comp_c (cdp, 09/2020)
    """

    # The search in array full_ind["row"] will be performed so flag check \
    # [fluid_comp_ref.identifier][fluid_comp.identifier]["row"] can be set to True. (cdp, 09/2020)
    check[fluid_comp_ref.identifier][fluid_comp_r.identifier].update(row=True)

    # search for all the values that are equal to ch_index in array \
    # full_ind["row"], excluding the index of the direct contact region, and \
    # store the corresponding indices (cdp, 09/2020)
    ind_found = (
        np.nonzero(full_ind["row"][boundary["variable_lower"] :] == ch_index)[0]
        + boundary["variable_lower"]
    )
    if len(ind_found) == 0:
        if (
            check[fluid_comp_ref.identifier][fluid_comp_r.identifier]["col"]
            == False
        ):
            # Search if there is some value equal to ch_index in the array \
            # full_ind["col"] calling function search_on_ind_col (cdp, 09/2020)
            total_connections_counter = search_on_ind_col(
                conductor,
                ch_index,
                full_ind,
                dict_topology,
                fluid_comp_ref,
                fluid_comp_r,
                check,
                already,
                total_connections_counter,
                boundary,
            )
        elif (
            check[fluid_comp_ref.identifier][fluid_comp_r.identifier]["col"] == True
        ):
            # no channel with larger channel index is connected with ch_index \
            # (cdp, 09/2020)
            print(f"No other channels are connected to {fluid_comp_r.identifier}\n")
    elif len(ind_found) > 0:
        # there is at least one channel with larger channel index that is \
        # connected with ch_index (cdp, 09/2020)
        # update key "variable_lower": the next search is done from this index \
        # value up to the end of arrays full_ind["row"] and full_ind["col"] \
        # (cdp, 09/2020)
        boundary.update(variable_lower=ind_found[0] + 1)
        # get the index of the linked channel(s) (row -> col) (cdp, 09/2020)
        ind_link = full_ind["col"][ind_found[0 : len(ind_found)]]
        # loop linked channels (cdp, 09/2020)
        jj = -1
        while jj < len(ind_link) - 1:
            jj = jj + 1
            ch_index = ind_link[jj]
            # get channel (cdp, 09/2020)
            fluid_comp_c = conductor.inventory.fluids.collection[ch_index]
            if (
                dict_topology[fluid_comp_ref.identifier].get(
                    fluid_comp_c.identifier
                )
                != None
                and f"{fluid_comp_c.identifier}_{fluid_comp_r.identifier}"
                in dict_topology[fluid_comp_ref.identifier][fluid_comp_c.identifier]
            ):
                # Skip operations since the connection is already preformed \
                # (cdp, 09/2020)
                jj = jj + 1
            else:
                if (
                    check[fluid_comp_ref.identifier].get(fluid_comp_c.identifier)
                    == None
                ):
                    # Update dictionary check: add key fluid_comp_c.identifier (cdp, 09/2020)
                    check[fluid_comp_ref.identifier][
                        fluid_comp_c.identifier
                    ] = dict(row=False, col=False)
                # construct channels link (cdp, 09/2020)
                if (
                    dict_topology[fluid_comp_ref.identifier].get(
                        fluid_comp_r.identifier
                    )
                    == None
                ):
                    # key fluid_comp_c.identifier does not exist, so it is added to the dictionary \
                    # and a list is created (cdp, 09/2020)
                    dict_topology[fluid_comp_ref.identifier][
                        fluid_comp_r.identifier
                    ] = [f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"]
                    print(
                        dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ][-1]
                        + "\n"
                    )
                    # update total_connections_counter (cdp, 09/2020)
                    total_connections_counter = total_connections_counter + 1
                else:
                    # key fluid_comp_c.identifier exists (cdp, 09/2020)
                    if (
                        f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                        not in dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ]
                    ):
                        # List is updated only if the contact identifier is not already in the \
                        # list (cdp, 09/2020)
                        dict_topology[fluid_comp_ref.identifier][
                            fluid_comp_r.identifier
                        ].append(
                            f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
                        )
                        print(
                            dict_topology[fluid_comp_ref.identifier][
                                fluid_comp_r.identifier
                            ][-1]
                            + "\n"
                        )
                        # update total_connections_counter (cdp, 09/2020)
                        total_connections_counter = total_connections_counter + 1
                # end if dict_topology[fluid_comp_ref.identifier].get(fluid_comp_c.identifier) == None \
                # (cdp, 09/2020)
                # find the index in array already["no"] of the element that must be \
                # deleted (cdp, 09/2020)
                i_del = np.nonzero(already["no"] == ch_index)[0]
                # update dictionary already (cdp, 09/2020)
                already["no"] = np.delete(already["no"], i_del, 0)
                if ch_index not in already["yes"]:
                    already["yes"][already["ii_yes"]] = ch_index
                    already.update(ii_yes=already["ii_yes"] + 1)
                if (
                    fluid_comp_c
                    not in dict_topology[fluid_comp_ref.identifier]["Group"]
                ):
                    # Add channel fluid_comp_c to list Group (cdp, 09/2020)
                    dict_topology[fluid_comp_ref.identifier]["Group"].append(
                        fluid_comp_c
                    )
                # call function search_on_ind_col to search if channel fluid_comp_c is \
                # linked to other channels (cdp, 09/2020)
                total_connections_counter = search_on_ind_col(
                    conductor,
                    ch_index,
                    full_ind,
                    dict_topology,
                    fluid_comp_ref,
                    fluid_comp_c,
                    check,
                    already,
                    total_connections_counter,
                    boundary,
                )
            # end if dict_topology[fluid_comp_ref.identifier].get(fluid_comp_c.identifier) != None and \
            # f"{fluid_comp_c.identifier}_{fluid_comp_r.identifier}" in dict_topology[fluid_comp_ref.identifier][fluid_comp_c.identifier] \
            # (cdp, 09/2020)
        # end while (cdp, 09/2020)
    # end if len(ind_found) (cdp, 09/2020)
    return total_connections_counter

# end function search_on_ind_row (cdp, 09/2020)


def get_thermal_contact_channels(
    conductor, rr, cc, fluid_comp_r, fluid_comp_c, flag_found
):

    """
    Function that recognize if there are some channels that are only in thermal contact with other channels. If one or both of the two considered channels belongs also to two different groups of channels in hydraulic parallel, they are not included in the list called Group, however the thermal contact is indicated in a suitable key-value pair. This is because channels that have both the properties of being in thermal contact and in hydraulic parallel should be threated considering the latter, while flow initialization is performed. (cdp, 09/2020)
    """

    if (
        conductor.dict_topology["ch_ch"]["Thermal_contact"].get(fluid_comp_r.identifier)
        == None
    ):
        # Update dictionary conductor.dict_topology["ch_ch"]["Thermal_contact"] \
        # (cdp, 09/2020)
        # key fluid_comp_r.identifier does not already exist (cdp, 09/2020)
        conductor.dict_topology["ch_ch"]["Thermal_contact"][
            fluid_comp_r.identifier
        ] = dict(Group=list(), Number=0, Actual_number=0)
    if (
        conductor.coupling.open_perimeter_fraction[
            fluid_comp_r.identifier, fluid_comp_c.identifier
        ]
        == 0.0
    ):
        # There is only thermal contact between fluid_comp_r and fluid_comp_c \
        # (cdp, 09/2020)
        # Update dictionary conductor.dict_topology["ch_ch"]["Thermal_contact"] \
        # (cdp, 09/2020)
        conductor.dict_topology["ch_ch"]["Thermal_contact"][fluid_comp_r.identifier][
            fluid_comp_c.identifier
        ] = f"{fluid_comp_r.identifier}_{fluid_comp_c.identifier}"
        if len(list(conductor.dict_topology["ch_ch"]["Hydraulic_parallel"].keys())) > 0:
            # There is at least one group of channels in hydraulic parallel \
            # (cdp, 09/2020)
            if flag_found.get(fluid_comp_r.identifier) == None:
                # initialize key fluid_comp_r.identifier of dictionary flag_found to False \
                # only once (cdp, 09/2020)
                flag_found[fluid_comp_r.identifier] = False
            # initialize key fluid_comp_c.identifier of dictionary flag_found to False for \
            # each value of cc (cdp, 09/2020)
            flag_found[fluid_comp_c.identifier] = False
            # Following lines check if one of this two channels belongs to a
            # group of channels in hydraulic parallel. In this case only the
            # one that does not belong to the group is added to the list of
            # channels with only thermal contact. (cdp, 09/2020)
            for fluid_comp_ref in list(
                conductor.dict_topology["ch_ch"]["Hydraulic_parallel"].keys()
            ):
                # Search if fluid_comp_r and fluid_comp_c are already inserted into two
                # different groups of channels in parallel. (cdp, 09/2020)
                if (
                    flag_found[fluid_comp_r.identifier] == False
                    and fluid_comp_r
                    in conductor.dict_topology["ch_ch"]["Hydraulic_parallel"][
                        fluid_comp_ref
                    ]["Group"]
                ):
                    # fluid_comp_r is in found in a group of channels in hydraulic \
                    # parallel (cdp, 09/2020)
                    flag_found[fluid_comp_r.identifier] = True
                # end if cc == rr + 1 (cdp, 09/2020)
                if (
                    flag_found[fluid_comp_c.identifier] == False
                    and fluid_comp_c
                    in conductor.dict_topology["ch_ch"]["Hydraulic_parallel"][
                        fluid_comp_ref
                    ]["Group"]
                ):
                    # fluid_comp_c is in found in a group of channels in hydraulic \
                    # parallel (cdp, 09/2020)
                    flag_found[fluid_comp_c.identifier] = True
            # end for fluid_comp_ref (cdp, 09/2020)
            if flag_found[fluid_comp_r.identifier] == False:
                # fluid_comp_r is not in hydraulic parallel with other channels \
                # (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"][
                    fluid_comp_r.identifier
                ]["Group"].append(fluid_comp_r)
            if flag_found[fluid_comp_c.identifier] == False:
                # fluid_comp_c is not in hydraulic parallel with other channels
                # (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"][
                    fluid_comp_r.identifier
                ]["Group"].append(fluid_comp_c)
        else:
            # there are no groups of channels in hydraulic parallel \
            # (cdp, 09/2020)
            if cc == rr + 1:
                # Add fluid_comp_r to the list Group only once (cdp, 09/2020)
                conductor.dict_topology["ch_ch"]["Thermal_contact"][
                    fluid_comp_r.identifier
                ]["Group"].append(fluid_comp_r)
            # Add channel fluid_comp_c to the list Group (cdp, 09/2020)
            conductor.dict_topology["ch_ch"]["Thermal_contact"][fluid_comp_r.identifier][
                "Group"
            ].append(fluid_comp_c)
        # end if len(list(conductor.dict_topology["ch_ch"]\
        # ["Hydraulic_parallel"].keys())) > 0 (cdp, 09/2020)
    # end if conductor.coupling.open_perimeter_fraction[rr, cc] == 0.0 (cdp, 09/2020)
    return flag_found

# end function get_thermal_contact_channels (cdp, 09/2020)


def get_conductor_interfaces(conductor: object, environment: object) -> None:
    """Function that identifies interfaces between conductor components, storing information in conductor attribute interface.

    Args:
        conductor (object): conductor object.
        environment (object): object with all the info that characterize the environment.

    Raises:
        ValueError: if jacket component is not of kind outer_insulation or wall_enclosure.
    """

    # Aliases
    fluid_components = conductor.inventory.fluids.collection
    solid_components = conductor.inventory.solids.collection
    interf_flag = conductor.coupling.contact_perimeter_flag

    # Namedtuple constructor definition.
    Interface_collection = namedtuple(
        "Interface_collection",
        (
            "fluid_fluid",
            "fluid_solid",
            "solid_solid",
            "env_solid",
        )
    )

    # Namedtuple constructor definition.
    Interface = namedtuple(
        "Interface",
        (
            "interf_name",
            "comp_1",
            "comp_2",
        )
    )

    # Namedtuple initialization: each field is an empty list to be filled
    # with interfaces.
    conductor.interface = Interface_collection(
        fluid_fluid=list(),
        fluid_solid=list(),
        solid_solid=list(),
        env_solid=list()
    )

    # Loop on FluidComponents.
    for f_comp_a_idx,f_comp_a in enumerate(fluid_components):
        # Loop on FluidComponents: identify fluid-fluid interfaces.
        for f_comp_b in fluid_components[f_comp_a_idx+1:]:
            # Check for interfaces.
            if (
                abs(
                    interf_flag[f_comp_a.identifier,f_comp_b.identifier]
                ) == 1
            ):
                # Build fluid-fluid interface.
                conductor.interface.fluid_fluid.append(
                    Interface(
                        interf_name=f"{f_comp_a.identifier}_{f_comp_b.identifier}",
                        comp_1=f_comp_a, # shallow copy: no waste of memory!
                        comp_2=f_comp_b, # shallow copy: no waste of memory!
                    )
                )
        # Loop on SolidComponent: identify fluid-solid interfaces.
        for s_comp in solid_components:
            # Check for iterfaces.
            if (
                abs(
                    interf_flag[f_comp_a.identifier,s_comp.identifier]
                ) == 1
            ):
                # Build fluid-solid interface.
                conductor.interface.fluid_solid.append(
                    Interface(
                        interf_name=f"{f_comp_a.identifier}_{s_comp.identifier}",
                        comp_1=f_comp_a, # shallow copy: no waste of memory!
                        comp_2=s_comp, # shallow copy: no waste of memory!
                    )
                )
    # Loop on SolidComponent.
    for s_comp_a_idx,s_comp_a in enumerate(solid_components):
        # Loop on SolidComponent: identify solid-solid interfaces.
        for s_comp_b in solid_components[s_comp_a_idx+1:]:
            # Check for interfaces.
            if (
                abs(
                    interf_flag[s_comp_a.identifier,s_comp_b.identifier]
                ) == 1
            ):
                # Build solid-solid interface.
                conductor.interface.solid_solid.append(
                    Interface(
                        interf_name=f"{s_comp_a.identifier}_{s_comp_b.identifier}",
                        comp_1=s_comp_a, # shallow copy: no waste of memory!
                        comp_2=s_comp_b, # shallow copy: no waste of memory!
                    )
                )
        # Check for environment-solid interfaces.
        if (
            abs(
                interf_flag[environment.KIND,s_comp_a.identifier]
            ) == 1
        ):
            # Check on jacket kind.
            if (
                s_comp_a.inputs.jacket_kind == "outer_insulation"
                or s_comp_a.inputs.jacket_kind == "whole_enclosure"
            ):
                # Build env-solid interface.
                conductor.interface.env_solid.append(
                    Interface(
                        interf_name=f"{environment.KIND}_{s_comp_a.identifier}",
                        # shallow copy: no waste of memory!
                        comp_1=environment,
                        comp_2=s_comp_a, # shallow copy: no waste of memory!
                    )
                )
            else:
                # Raise error
                raise ValueError(f"JacketComponent of kind {s_comp_a.inputs.jacket_kind} can not have and interface with the environment.\n"
                )
