import warnings
import numpy as np
from utility_functions.auxiliary_functions import get_from_xlsx
from hydraulics.hydraulic_flags import HydraulicBC, FlowDirection


def gen_flow(cond):

    """
    ##############################################################################
    Function that generate the initial flow in the conductor according to the initial values of inlet temperature, inlet and or outlet pressure and inlet mass flow rate, as ruled by flag INTIAL.
    ##############################################################################
    """

    # FIXED PARAMETERS FOR OPERATING
    Max_iter = 1000  # maximum allowed number of iterations (cdp, 09/2020)
    tol = 1.0e-10  # required tolerance (cdp, 09/2020)
    totChannelCrossSection = 0
    path = cond.file_paths.external_flow

    # (cdp, 07/2020)
    for fluid_comp in cond.inventory.fluids.collection:
        totChannelCrossSection = (
            totChannelCrossSection + fluid_comp.channel.inputs.cross_section
        )
    # Compute crossFraction for each fluid_comp (cdp, 07/2020)
    for fluid_comp in cond.inventory.fluids.collection:
        fluid_comp.crossFraction = (
            fluid_comp.channel.inputs.cross_section / totChannelCrossSection
        )

    # Call function Get_flow_no_hydraulic_parallel_channels to evaluate initial \
    # flow parameters of isolated (not in hydraulic parallel) channels \
    # (cdp, 09/2020)
    get_flow_no_hydraulic_parallel_channels(cond, path, Max_iter, tol)
    # Call function Get_flow_hydraulic_parallel_channels to evaluate initial \
    # flow parameters of channels in hydraulic parallel, i.e. constituting \
    # an interface and therefore having same inlet and outlet pressure \
    # (cdp, 09/2020)
    get_flow_hydraulic_parallel_channels(cond, path, Max_iter, tol)
    # Call function Get_inlet_conductor_mfr to evaluate total conductor inlet \
    # mass flow rate and channels flow fractions (cdp, 09/2020)
    get_inlet_conductor_mfr(cond)


def get_flow_no_hydraulic_parallel_channels(cond, path, Max_iter, tol):

    """
    Function that evaluate initial fluid properties for channels that are not in hydraulic parallel, according to the fluid type and the initial contition given by the flag INTIAL. It is allowed to use different values of flag INTIAL if there is more than one stand alone channel. (cdp, 09/2020)
    """

    # Channels that are in thermal contact with other channels and do not also \
    # belong to a group of channels in hydraulic parallel (cdp, 09/2020)
    for fluid_comp_ref in cond.dict_topology["ch_ch"]["Thermal_contact"].keys():
        for fluid_comp in cond.dict_topology["ch_ch"]["Thermal_contact"][
            fluid_comp_ref
        ]["Group"]:
            # Call function Initialize_flow_no_hydraulic_parallel to initialize \
            # flow of channel fluid_comp (cdp, 09/2020)
            initialize_flow_no_hydraulic_parallel(cond, fluid_comp, path, Max_iter, tol)
        # end for fluid_comp (cdp, 09/2020)
    # end for fluid_comp_ref (cdp, 09/2020)
    # Channels that are not in thermal contact and not in hydraulic parallel \
    # (cdp, 09/2020)
    for fluid_comp in cond.dict_topology["Standalone_channels"]:
        # Call function Initialize_flow_no_hydraulic_parallel to initialize flow \
        # of channel fluid_comp (cdp, 09/2020)
        initialize_flow_no_hydraulic_parallel(cond, fluid_comp, path, Max_iter, tol)
    # end for fluid_comp


# end function Get_flow_no_hydraulic_parallel_channels


def initialize_flow_no_hydraulic_parallel(cond, fluid_comp, path, Max_iter, tol):

    """
    Function that actually initializes the flow for channels that are not in hydraulic parallel (cdp, 09/2020)
    """

    if fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP:
        if (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP and not fluid_comp.coolant.operations.bc_values_from_file):
            # inlet pressure (cdp, 06/2020)
            p_inl = fluid_comp.coolant.operations.inlet_pressure
            # outlet pressure (cdp, 06/2020)
            p_out = fluid_comp.coolant.operations.outlet_pressure
            # inlet temperature (cdp, 06/2020)
            T_inl = fluid_comp.coolant.operations.inlet_temperature
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, 
        INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following 
        flow input from Worksheet CHAN of file \
        {cond.file_paths.operation_path} parameters:\nPREINL = {p_inl} Pa;
        \nPREOUT = {p_out} Pa;\nTEMINL = {T_inl} K.\n"""
            )
        elif (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP and fluid_comp.coolant.operations.bc_values_from_file):
            # call get_from_xlsx
            [flow_par, flagSpecfield] = get_from_xlsx(
                cond,
                path,
                fluid_comp,
                "INTIAL",
                -int(fluid_comp.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"""flagSpecfield == {flagSpecfield}: still to be decided if 
            it is useful and if yes still to be defined\n"""
            )
            T_inl = flow_par[0]
            p_inl = flow_par[2]
            p_out = flow_par[3]
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, 
        INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following 
        flow input parameters from Worksheet CHAN of file flow_dummy.xlsx 
        parameters:\nPREINL = {p_inl} Pa;\nPREOUT = {p_out} Pa;
        \nTEMINL = {T_inl} K.\n"""
            )
        p_ave = (p_inl + p_out) / 2
        # Invoke method eval_coolant_density_din_viscosity_gen_flow to evaluate density and dynamic viscosity at p_ave and T_inl
        rho, mu = fluid_comp.coolant.eval_coolant_density_din_viscosity_gen_flow(
            [p_ave], [T_inl]
        )
        # SET INITIAL CONDITIONS FOR THE ITERATION
        dpin = 0.0
        dpout = 0.0
        delta_p = (p_inl - dpin) - (p_out + dpout)
        # old version (before introduction of flag flow_dir)
        # if p_inl > p_out:
        #   delta_p = (p_inl - dpin) - (p_out + dpout)
        # elif p_out > p_inl:
        #   delta_p = (p_out - dpout) - (p_inl + dpin)
        if delta_p < 0:
            raise ValueError(
                f"""Error in function {gen_flow.__name__}!\nNegative 
      pressure drop: delta_p = {delta_p}"""
            )
        # Compute velocity invoking method compute_velocity_gen_flow
        velocity = fluid_comp.coolant.compute_velocity_gen_flow(
            cond.inputs.zlength,
            fluid_comp.channel,
            Max_iter,
            delta_p,
            rho,
            mu,
            tol,
        )
        # DETERMINE THE MASSFLOW (WITH SIGN !)
        # The float is introduced to convert the result from np array of shape \
        # (1,) to a scalar float (cdp, 09/2020)
        mdot_inl = float(
            fluid_comp.coolant.compute_mass_flow_with_direction(
                fluid_comp.channel.flow_sign, rho, velocity
            )
        )
        if abs(mdot_inl) != fluid_comp.coolant.operations.inlet_mass_rate:
            warnings.warn(
                f"Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluatedinlet mass flow rate is different from the one in Worksheet CHAN of input file {cond.file_paths.operation_path}: {abs(mdot_inl)} != {fluid_comp.coolant.operations.inlet_mass_rate}. This value is overwritten by the evaluated one with the correct sign according to the flow direction:\nMDTIN = {mdot_inl} kg/s\n"
            )
            # overwriting fluid_comp inlet mass flow rate (cdp, 06/2020)
            fluid_comp.coolant.operations.inlet_mass_rate = mdot_inl
    # end abs(INTIAL == 1): output mdot_inl
    elif fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY:
        if (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY and not fluid_comp.coolant.operations.bc_values_from_file):
            # Inlet mass flow rate (cdp, 08/2020)
            mdot_out = fluid_comp.coolant.operations.outlet_mass_rate
            # Inlet pressure (cdp, 08/2020)
            p_inl = fluid_comp.coolant.operations.inlet_pressure
            # Inlet temperature (cdp, 08/2020)
            T_inl = fluid_comp.coolant.operations.inlet_temperature
            # Inlet temperature (cdp, 09/2020)
            T_out = fluid_comp.coolant.operations.outlet_temperature
            warnings.warn(
                f"""Function {gen_flow}, {fluid_comp.identifier}, 
                INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input from Worksheet CHAN of file {cond.file_paths.operation_path} parameters:\nPREINL = {p_inl} Pa;\nTEMINL = {T_inl} K;\nMDTOUT = {mdot_out}.\n"""
                )
        elif (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY and fluid_comp.coolant.operations.bc_values_from_file):
            # All values form flow_dummy.xlsx (cdp, 07/2020)
            [flow_par, flagSpecfield] = get_from_xlsx(
                cond,
                path,
                fluid_comp,
                "INTIAL",
                -int(fluid_comp.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"flagSpecfield == {flagSpecfield}: still to be decided if it useful and if yes still to be defined\n"
            )
            # Outlet mass flow rate (cdp, 08/2020)
            mdot_out = flow_par[3]
            # Inlet pressure (cdp, 08/2020)
            p_inl = flow_par[2]
            # Inlet temperature (cdp, 08/2020)
            T_inl = flow_par[0]
            # Outlet temperature (cdp, 09/2020)
            T_out = flow_par[1]
            warnings.warn(
                f"""Function {gen_flow}, {fluid_comp.identifier}, 
                    INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input parameters from Worksheet CHAN of file flow_dummy.xlsx parameters:\nPREINL = {p_inl} Pa;\nTEMINL = {T_inl} K;\nMDTOUT = {mdot_out}.\n"""
                )
        get_missing_pressure_no_hydraulic_parallel(
            cond,
            fluid_comp,
            mdot_out,
            p_inl,
            T_inl,
            T_out,
            Max_iter,
            tol,
            intial=fluid_comp.coolant.operations.hydraulic_bc_type,
        )
    # end abs(INTIAL == 2): output p_out
    elif fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE:
        if (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE and not fluid_comp.coolant.operations.bc_values_from_file):
            # inlet temperature (cdp, 06/2020)
            T_inl = fluid_comp.coolant.operations.inlet_temperature
            # outlet pressure (cdp, 07/2020)
            p_out = fluid_comp.coolant.operations.outlet_pressure
            # inlet mass flow rate (cdp, 07/2020)
            mdot_inl = fluid_comp.coolant.operations.inlet_mass_rate
            # Inlet temperature (cdp, 09/2020)
            T_out = fluid_comp.coolant.operations.outlet_temperature
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, 
                    INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input from Worksheet CHAN of file {cond.file_paths.operation_path} parameters:\nPREOUT = {p_out} Pa;\nTEMINL = {T_inl} K;\nMDTIN = {mdot_inl}.\n"""
            )
        elif (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE and fluid_comp.coolant.operations.bc_values_from_file):
            # all values from flow_dummy.xlsx: call get_from_xlsx (cdp, 07/2020)
            [flow_par, flagSpecfield] = cond.Get_from_xlsx(
                path,
                fluid_comp,
                0.0,
                "INTIAL",
                -int(fluid_comp.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"flagSpecfield == {flagSpecfield}: still to be decided if it useful and if yes still to be defined\n"
            )
            # inlet temperature (cdp, 07/2020)
            T_inl = flow_par[0]
            # outlet pressure (cdp, 07/2020)
            p_out = flow_par[2]
            # inlet mass flow rate (cdp, 07/2020)
            mdot_inl = flow_par[3]
            # Outlet temperature (cdp, 09/2020)
            T_out = flow_par[1]
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, 
                    INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input parameters from Worksheet CHAN of file flow_dummy.xlsx parameters:\nPREOUT = {p_out} Pa;\nTEMINL = {T_inl} K;\nMDTIN = {mdot_inl}.\n"""
            )
        get_missing_pressure_no_hydraulic_parallel(
            cond,
            fluid_comp,
            mdot_inl,
            p_out,
            T_inl,
            T_out,
            Max_iter,
            tol,
            intial=fluid_comp.coolant.operations.hydraulic_bc_type,
        )


# end function Initialize_flow_no_hydraulic_parallel (cdp, 09/2020)


def get_missing_pressure_no_hydraulic_parallel(
    cond, fluid_comp, mdot_known, p_known, T_inl, T_out, Max_iter, tol, intial=2
):

    """
    Function that evaluate outlet pressure if abs(INTIAL) = 2 or inlet pressure if abs(INTIAL) = 3 for stand alone channel(s). The algorithm is almost the same in this two cases and this is why this function is created. (cdp, 09/2020)
    ----------------------------------------------------------------------------
    ** RECIPE TO FIND OUTLET PRESSURE, CASE abs(INTIAL) == 2 **
    1) guess initial pressure drop evaluated with inlet pressure and temperature and outlet mass flow rate
    2) evaluate p_out as p_inl - delta_p_old
    3) evaluate average pressure, (p_inl + p_out)/2
    4) evaluate average temperature, (T_inl + T_out)/2
    5) evaluate density and viscosity at average pressure and temperature
    6) evaluate Re number as mdot*D/(A*mu) where mu is the viscosity evaluated above
    7) exploiting Re, evaluate friction factor for the given geometry
    8) evaluate the new pressure drop as:
       delta_p = l/(2*D*A^2)*f(Re)*rho(p_ave,T_ave)*mdot^2
    9) evaluate relative error on pressure drop
    10) set delta_p_old equal to delta_p_new
    11) iterate starting from step 2) until error >= tol and iter < Max_iter.
    ----------------------------------------------------------------------------
    ** RECIPE TO FIND INLET PRESSURE, CASE abs(INTIAL) == 3 **
    1) guess initial pressure drop evaluated with outlet pressure and inlet mass flow rate and temperature
    2) evaluate p_inl as p_out + delta_p_old
    3) evaluate average pressure, (p_inl + p_out)/2
    4) evaluate average temperature, (T_inl + T_out)/2
    5) evaluate density and viscosity at average pressure and temperature
    6) evaluate Re number as mdot*D/(A*mu) where mu is the viscosity evaluated above
    7) exploiting Re, evaluate friction factor for the given geometry
    8) evaluate the new pressure drop as:
       delta_p = l/(2*D*A^2)*f(Re)*rho(p_ave,T_ave)*mdot^2
    9) evaluate relative error on pressure drop
    10) set delta_p_old equal to delta_p_new
    11) iterate starting from step 2) until error >= tol and iter < Max_iter.
    ----------------------------------------------------------------------------
    """

    list_words = ["inlet", "outlet"]
    list_symbols = ["p_inl", "p_out"]
    list_keys = ["inlet_pressure", "outlet_pressure"]

    # geometry coefficient, Fanning friction factor considered (cdp, 09/2020)
    g0 = (
        2.0
        * cond.inputs.zlength
        / (
            fluid_comp.channel.inputs.hydraulic_diameter
            * (fluid_comp.channel.inputs.cross_section ** 2)
        )
    )
    # Invoke method eval_coolant_density_din_viscosity_gen_flow to evaluate density and dynamic viscosity at known pressure and inlet temperature
    (
        rho_known,
        mu_known,
    ) = fluid_comp.coolant.eval_coolant_density_din_viscosity_gen_flow(
        [p_known], [T_inl]
    )
    # Invoke method eval_reynolds_from_mass_flow_rate to evaluate Reynolds number at known pressure and inlet temperature
    Re_known = fluid_comp.coolant.eval_reynolds_from_mass_flow_rate(mdot_known, mu_known)
    # Friction factor at inlet pressure and temperature, nodal = None specities that total friction factor in Gen_Flow module is evaluated (cdp, 09/2020)
    fluid_comp.channel.eval_friction_factor(np.array([Re_known]), nodal=None)
    # Pressure drop evaluated with properties at inlet, to be able to deal \
    # with any fluid type (cdp, 09/2020)
    # Conversion to scalar via item() since the operands are one-element
    # arrays (float() on 1-d arrays raises TypeError with numpy >= 1.25).
    delta_p_old = (
        g0
        * fluid_comp.channel.friction_factors[None].total
        * mdot_known ** 2
        / rho_known
    ).item()  # Pa
    T_ave = (T_inl + T_out) / 2  # average temperature (cdp, 09/2020)
    err_delta_p = 10.0  # error initialization
    iteration = 0
    while err_delta_p >= tol and iteration < Max_iter:
        iteration = iteration + 1
        if abs(intial) == 2:
            # Evaluate outlet pressure (cdp, 09/2020)
            p_missing = p_known - delta_p_old
        elif abs(intial) == 3:
            # Evaluate inlet pressure (cdp, 09/2020)
            p_missing = p_known + delta_p_old
        p_ave = (p_known + p_missing) / 2.0  # average pressure (cdp, 09/2020)
        # Invoke method eval_coolant_density_din_viscosity_gen_flow to evaluate density and dynamic viscosity at average pressure and temperature
        (
            rho_ave,
            mu_ave,
        ) = fluid_comp.coolant.eval_coolant_density_din_viscosity_gen_flow(
            [p_ave], [T_ave]
        )
        # Invoke method eval_reynolds_from_mass_flow_rate to evaluate Reynolds number at average pressure and temperature
        Re_ave = fluid_comp.coolant.eval_reynolds_from_mass_flow_rate(mdot_known, mu_ave)
        # Friction factor at average pressure and temperature, nodal = None specities that total friction factor in Gen_Flow module is evaluated (cdp, 09/2020)
        fluid_comp.channel.eval_friction_factor(Re_ave, nodal=None)
        # New pressure drop evaluation: conversion to float is necessary \
        # to avoid TypeError when call function dhe and vische after the \
        # first iteration (cdp, 09/2020)
        delta_p_new = (
            g0
            * fluid_comp.channel.friction_factors[None].total
            * mdot_known ** 2
            / rho_ave
        ).item()  # Pa
        err_delta_p = abs(delta_p_old - delta_p_new) / delta_p_old
        delta_p_old = delta_p_new
    # end while
    if abs(intial) == 2:
        word = list_words[1]
        symbol = list_symbols[1]
        key = list_keys[1]
    elif abs(intial) == 3:
        word = list_words[0]
        symbol = list_symbols[0]
        key = list_keys[0]
    if err_delta_p >= tol and iteration >= Max_iter:
        warnings.warn(
            f"""INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}.\nRequired tolerance not reached after {iteration} iterations: {err_delta_p} >= {tol}.\nEvaluated {word} pressure:\n
      {symbol} = {p_missing} bar"""
        )
    if p_missing != getattr(fluid_comp.coolant.operations, key):
        warnings.warn(
            f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
      {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated {word} pressure is 
      different from the one in Worksheet CHAN of input file 
      {cond.file_paths.operation_path}:
      {p_missing} != {getattr(fluid_comp.coolant.operations, key)}.\n
      This value is overwritten by the evaluated one:\n
      {symbol} = {p_missing} Pa\n"""
        )
        # overwriting channel missing pressure (cdp, 09/2020)
        setattr(fluid_comp.coolant.operations, key, p_missing)
    # Assign the correct sign to the mass flow rate according to the flow direction (always positive in the input file)
    fluid_comp.coolant.operations.inlet_mass_rate = (
        fluid_comp.channel.flow_sign * fluid_comp.coolant.operations.inlet_mass_rate
    )


# end function Get_missing_pressure_no_hydraulic_parallel (cdp, 09/2020)


def get_flow_hydraulic_parallel_channels(cond, path, Max_iter, tol):

    """
    Function that evaluates initial fluid properties for channels constitute an hydraulic parallel and thus share the same inlet and outlet pressure. This is dove keeping into account the fluid type and the initial contitions for each channels, given by the flag INTIAL. Absolute value of INTIAL must be the same for all the channels that constitute the interface, otherwise and error is raised. (cdp, 09/2020)
    """

    # Call function Check_INTIAL_values to check that abs(INTIAL) is the same \
    # for all the channels that are in contact (cdp, 09/2020)
    check_intial_values(cond)
    # Loop on channel groups that are in hydraulic parallel (cdp, 09/2020)
    for key in cond.dict_topology["ch_ch"]["Hydraulic_parallel"].keys():
        # Remember that all channels in a group have the same absolute value of \
        # flag INTIAL and that the reference channel is the first in the list of \
        # objects "Group" (cdp, 09/2020)
        chan_group = cond.dict_topology["ch_ch"]["Hydraulic_parallel"][key]["Group"]
        # Number of channels costituting the group (cdp, 09/2020)
        N_group = cond.dict_topology["ch_ch"]["Hydraulic_parallel"][key]["Number"]
        INTIAL_ref = int(chan_group[0].coolant.operations.hydraulic_bc_type)
        if INTIAL_ref == 1:
            # Call function Abs_INTIAL_equal_1_hp to initialize flow parameters for \
            # channel groups characterized by abs(INTIAL) = 1 (cdp, 09/2020)
            abs_intial_equal_1_hp(cond, chan_group, N_group, path, Max_iter, tol)
        elif INTIAL_ref == 2:
            # Call function Abs_INTIAL_equal_2_or_3_hp to initialize flow parameters \
            # for channel groups characterized by abs(INTIAL) = 2 (cdp, 09/2020)
            abs_intial_equal_2_or_3_hp(
                cond, chan_group, N_group, path, tol, intial=INTIAL_ref
            )
        elif INTIAL_ref == 3:
            # Call function Abs_INTIAL_equal_2_or_3_hp to initialize flow parameters \
            # for channel groups characterized by abs(INTIAL) = 3 (cdp, 09/2020)
            abs_intial_equal_2_or_3_hp(
                cond, chan_group, N_group, path, tol, intial=INTIAL_ref
            )
        else:
            # Raise error (cdp, 09/2020)
            raise ValueError(
                f"ERROR: not valid INTIAL value abs(INTIAL) = {INTIAL_ref}\n"
            )


# end function Get_flow_hydraulic_parallel_channels


def check_intial_values(cond):

    """
    Function that checks if the absolute value of flag INTIAL is the same for all channels that constitute an interface. It raises an error if this condition is not True. The check is performed on each group of channels that are in contact to stop the initialization calculations before performing them for the correctly initialized channel groups and then having to repeat them if an error is found for another channel group. (cdp, 09/2020)
    """

    # Dictionary declaration: it is explotited to raise and write the error \
    # message (cdp, 09/2020)
    dict_raise_error = dict()
    for key in cond.dict_topology["ch_ch"]["Hydraulic_parallel"].keys():
        # key is the ID of the first channel constituting the interface \
        # (cdp, 09/2020)
        # Reference value for flag INTIAL is the absolute value of INTIAL \
        # assigned to the first channel constititing the interface (cdp, 09/2020)
        INTIAL_ref = int(
            cond.dict_topology["ch_ch"]["Hydraulic_parallel"][key]["Group"][
                0
            ].coolant.operations.hydraulic_bc_type
        )
        dict_raise_error[key] = dict(
            channels=list(), flag_value=list(), reference=INTIAL_ref
        )
        for fluid_comp in cond.dict_topology["ch_ch"]["Hydraulic_parallel"][key][
            "Group"
        ][1:]:
            if int(fluid_comp.coolant.operations.hydraulic_bc_type) != INTIAL_ref:
                # Fill the list with channel ID that have a different value of intial \
                # wrt to the reference one (cdp, 09/2020)
                dict_raise_error[key]["channels"].append(fluid_comp.identifier)
                dict_raise_error[key]["flag_value"].append(
                    fluid_comp.coolant.operations.hydraulic_bc_type
                )
            if len(dict_raise_error[key]["channels"]) == 0:
                # In this case the list is empty, i.e. all the channels have the
                # correct value of flag INTIAL, so this key can be removed from the \
                # dictionary (cdp, 09/2020)
                dict_raise_error.pop(key)
    if len(dict_raise_error) > 0:
        # There is at least one not empty list so an error must be raised \
        # (cdp, 09/2020)
        for key in dict_raise_error:
            print("---------------------------------------------------------------\n")
            print(
                f"Reference channel: {key};\nreference INTIAL value: {dict_raise_error[key]['reference']}\nChannels with incorrect value:\n"
            )
            for ii in range(len(dict_raise_error[key]["channels"])):
                print(
                    f"Channel identifier: {dict_raise_error[key]['channels'][ii]}; INTIAL: {dict_raise_error[key]['flag_value'][ii]} \n"
                )
            print("---------------------------------------------------------------\n")
        # raise error since INTIAL values are different for at least one channel
        raise ValueError(
            f"""ERROR: all the above listed channels have a different INTIAL value wrt the reference one. The absolute value of flag INTIAL for channels that are in contact must be equal to the reference value indicated above. User is invited to check and correct given values in sheet CHAN of file {cond.file_paths.operation_path}."""
        )


# end function Check_INTIAL_values (cdp, 09/2020)


def abs_intial_equal_1_hp(cond, chan_group, N_group, path, Max_iter, tol):

    """
    Function that evaluate initial flow parameters for a channel group in hydraulic parallel characterized by the same fluid and the same value of flag INTIAL: asb(INTIAL) = 1 in this case. (cdp, 09/2020)
    """

    # flow parameters initialization (cdp, 09/2020)
    p_inl = np.zeros(N_group)
    p_out = np.zeros(N_group)
    T_inl = np.zeros(N_group)
    flow_dir = list()
    # Loop on the channels of the group (cdp, 09/2020)
    for ii in range(N_group):
        fluid_comp = chan_group[ii]
        flow_dir.append(fluid_comp.coolant.operations.flow_direction)
        if (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP and not fluid_comp.coolant.operations.bc_values_from_file):
            # inlet pressure (cdp, 06/2020)
            p_inl[ii] = fluid_comp.coolant.operations.inlet_pressure
            # outlet pressure (cdp, 06/2020)
            p_out[ii] = fluid_comp.coolant.operations.outlet_pressure
            # inlet temperature (cdp, 06/2020)
            T_inl[ii] = fluid_comp.coolant.operations.inlet_temperature
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input 
        from Worksheet CHAN of file {cond.file_paths.operation_path} 
        parameters:
        \nPREINL = {p_inl[ii]} Pa;
        \nPREOUT = {p_out[ii]} Pa;
        \nTEMINL = {T_inl[ii]} K.\n"""
            )
        elif (fluid_comp.coolant.operations.hydraulic_bc_type is HydraulicBC.IMPOSE_PRESSURE_DROP and fluid_comp.coolant.operations.bc_values_from_file):
            # call get_from_xlsx
            [flow_par, flagSpecfield] = get_from_xlsx(
                cond,
                path,
                fluid_comp,
                "INTIAL",
                -int(fluid_comp.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"""flagSpecfield == {flagSpecfield}: still to be decided
            if it is useful and if yes still to be defined\n"""
            )
            # inlet pressure (cdp, 06/2020)
            p_inl[ii] = flow_par[2]
            # outlet pressure (cdp, 06/2020)
            p_out[ii] = flow_par[3]
            # inlet temperature (cdp, 06/2020)
            T_inl[ii] = flow_par[0]
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input 
        parameters from Worksheet CHAN of file flow_dummy.xlsx parameters:
        \nPREINL = {p_inl[ii]} Pa;
        \nPREOUT = {p_out[ii]} Pa;
        \nTEMINL = {T_inl[ii]} K.\n"""
            )
        # end if fluid_comp.coolant.operations["INTIAL"] (cdp, 09/2020)
    # end for ii (cdp, 09/2020)
    # Evaluate inlet and outlet average pressure with array smart notation \
    # (cdp, 09/2020)
    p_inl_ave = np.sum(p_inl) / N_group
    p_out_ave = np.sum(p_out) / N_group
    p_ave = (p_inl_ave + p_out_ave) / 2.0
    # Evaluate delta_p and flow direction (cdp, 09/2020)
    dpin = 0.0
    dpout = 0.0
    if flow_dir.count(FlowDirection.FORWARD) == len(flow_dir):
        delta_p = (p_inl_ave - dpin) - (p_out_ave + dpout)
        print("Forward flow\n")
    elif flow_dir.count(FlowDirection.BACKWARD) == len(flow_dir):
        delta_p = (p_inl_ave + dpin) - (p_out_ave - dpout)
        print("Backward flow\n")
    else:
        raise ValueError(
            "Channels in hydraulic parallel must all have the same flow direction.\nPlease, check flag FLOWDIR in sheet CHANNEL of input file conductor_operation.xlsx.\n"
        )
        # Old (to be expanded) delta_p = 0
        # warnings.warn(f"Pressure drop = 0\nChannels belonging to group {fluid_comp.identifier} do not participate to heat exchange\n")
    if delta_p < 0:
        raise ValueError(
            f"Error in function {gen_flow.__name__}!\nNegative pressure drop: delta_p = {delta_p}"
        )
    # Compute mass flow rate (cdp, 09/2020)
    for ii in range(N_group):
        fluid_comp = chan_group[ii]
        # Invoke method eval_coolant_density_din_viscosity_gen_flow to evaluate density and dynamic viscosity at average pressure and inlet Temperature
        (
            rho_inl,
            mu_inl,
        ) = fluid_comp.coolant.eval_coolant_density_din_viscosity_gen_flow(
            [p_ave], [T_inl[ii]]
        )
        # Compute velocity invoking method compute_velocity_gen_flow
        velocity = fluid_comp.coolant.compute_velocity_gen_flow(
            cond.inputs.zlength,
            fluid_comp.channel,
            Max_iter,
            delta_p,
            rho_inl,
            mu_inl,
            tol,
        )
        # DETERMINE THE MASSFLOW (WITH SIGN !)
        # The float is introduced to convert the result from np array of shape \
        # (1,) to a scalar float (cdp, 09/2020)
        mdot_inl = float(
            fluid_comp.coolant.compute_mass_flow_with_direction(
                fluid_comp.channel.flow_sign, rho_inl, velocity
            )
        )
        if abs(mdot_inl) != fluid_comp.coolant.operations.inlet_mass_rate:
            warnings.warn(
                f"Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated inlet mass flow rate is different from the one in Worksheet CHAN of input file {cond.file_paths.operation_path}: {abs(mdot_inl)} != {fluid_comp.coolant.operations.inlet_mass_rate}.\n This value is overwritten by the evaluated one with the correct sign according to the flow direction:\nMDTIN = {mdot_inl}\n"
            )
            # overwriting channel inlet mass flow rate (cdp, 06/2020)
            fluid_comp.coolant.operations.inlet_mass_rate = mdot_inl
        if p_inl_ave != fluid_comp.coolant.operations.inlet_pressure:
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated inlet pressure is different 
        from the one in Worksheet CHAN of input file 
        {cond.file_paths.operation_path}:
        {p_inl_ave} != {fluid_comp.coolant.operations.inlet_pressure}.\n
        This value is overwritten by the evaluated one:\n
        PREINL = {p_inl_ave}\n"""
            )
            # overwriting channel inlet pressure (cdp, 06/2020)
            fluid_comp.coolant.operations.inlet_pressure = p_inl_ave
        if p_out_ave != fluid_comp.coolant.operations.outlet_pressure:
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated outlet pressure is different 
        from the one in Worksheet CHAN of input file 
        {cond.file_paths.operation_path}:
        {p_out_ave} != {fluid_comp.coolant.operations.outlet_pressure}.\n
        This value is overwritten by the evaluated one:\n
        PREOUT = {p_out_ave}\n"""
            )
            # overwriting channel outlet pressure (cdp, 06/2020)
            fluid_comp.coolant.operations.outlet_pressure = p_out_ave
    # end for ii output MDTIN, p_inl_ave, p_out_ave


# end function Abs_INTIAL_equal_1_hp (cdp, 09/2020)


def abs_intial_equal_2_or_3_hp(cond, chan_group, N_group, path, tol, intial=2):

    """
    Function that evaluate initial flow parameters for a channel group in hydraulic parallel characterized by the same fluid and the same value of flag INTIAL: asb(INTIAL) = 2 or asb(INTIAL) = 3 in this case. This two cases are treated with the same function since they are similar; it should be remembered however that all the channels of a group are characterized by the same INTIAL value. (cdp, 09/2020)
    """

    list_words = ["inlet", "outlet"]
    list_symbols = ["p_inl", "p_out"]
    list_keys = ["inlet_pressure", "outlet_pressure", "inlet_mass_rate", "outlet_mass_rate"]

    # flow parameters initialization (cdp, 09/2020)
    mdot_known = np.zeros(N_group)
    p_known = np.zeros(N_group)
    T_inl = np.zeros(N_group)
    # Other usefull quantities initialization (cdp, 09/2020)
    g0 = np.zeros(N_group)

    if abs(intial) == 2:
        word = list_words[1]
        key_a = list_keys[0]
        key_b = list_keys[1]
        key_mfr = list_keys[3]
        symbol = list_symbols[1]
    elif abs(intial) == 3:
        word = list_words[0]
        key_a = list_keys[1]
        key_b = list_keys[0]
        key_mfr = list_keys[2]
        symbol = list_symbols[0]

    # Loop on the channels of the group (cdp, 09/2020)
    for ii in range(N_group):
        fluid_comp = chan_group[ii]
        if not fluid_comp.coolant.operations.bc_values_from_file:
            # Inlet mass flow rate (cdp, 08/2020)
            mdot_known[ii] = getattr(fluid_comp.coolant.operations, key_mfr)
            # Known pressure (cdp, 08/2020)
            p_known[ii] = getattr(fluid_comp.coolant.operations, key_a)
            # Inlet temperature (cdp, 08/2020)
            T_inl[ii] = fluid_comp.coolant.operations.inlet_temperature
            warnings.warn(
                f"""Function {gen_flow}, {fluid_comp.identifier}, INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input parameters from Worksheet CHAN of file {cond.file_paths.operation_path}:\n{key_a} = {p_known[ii]} Pa;\nTEMINL = {T_inl[ii]} K;\n{key_mfr} = {mdot_known[ii]} kg/s.\n"""
            )
        elif fluid_comp.coolant.operations.bc_values_from_file:
            # All values form flow_dummy.xlsx (cdp, 07/2020)
            [flow_par, flagSpecfield] = get_from_xlsx(
                cond,
                path,
                fluid_comp,
                "INTIAL",
                -int(fluid_comp.coolant.operations.hydraulic_bc_type),  # reconstruct the negative INTIAL flag (values from file)
            )
            print(
                f"flagSpecfield == {flagSpecfield}: still to be decided if it useful and if yes still to be defined\n"
            )
            # Inlet mass flow rate (cdp, 08/2020)
            mdot_known[ii] = flow_par[3]
            # Known pressure (cdp, 08/2020)
            p_known[ii] = flow_par[2]
            # Inlet temperature (cdp, 08/2020)
            T_inl[ii] = flow_par[0]
            warnings.warn(
                f"""Function {gen_flow}, {fluid_comp.identifier}, INTIAL == {fluid_comp.coolant.operations.hydraulic_bc_type.name}: you are imposing following flow input parameters from Worksheet CHAN of file flow_dummy.xlsx parameters:\n{key_a} = {p_known[ii]} Pa;\nTEMINL = {T_inl[ii]} K;\n{key_mfr} = {mdot_known[ii]} kg/s.\n"""
            )
        # end if INTIAL (cdp, 09/2020)
        # Evaluate channels geometry parameters g0 to compute outlet \
        # pressure, Fanning friction factor considered (cdp, 09/2020)
        g0[ii] = (
            2.0
            * cond.inputs.zlength
            / (
                fluid_comp.channel.inputs.hydraulic_diameter
                * (fluid_comp.channel.inputs.cross_section ** 2)
            )
        )
    # end for ii (cdp, 09/2020)

    # Algorithm description (cdp, 09/2020)
    # To compute pressure drop, the hydraulic characteristic of each channel is
    # determined, approximated with a parable having vertex in the origin of
    # the Cartesian reference frame (delta_p-mfr) and thus described by the
    # equation: delta_p_ch = alpha_ch*mdot_known**2
    # where alpha_ch is to be determined, exploiting the available information,
    # as: 
    # alpha_ch = g0*fric/rho
    # with alpha_ch known, the estimated channels pressure drop is given by:
    # delta_p_group = mdot_known_tot/sum(1/sqrt(alpha_ch))**2
    # Finally the actual mass flow rate repartition is evaluated, and the check
    # on mass conservation is done.
    # To evaluate delta_p properties are evaluated with the inlet temperature
    # and the average inlet or outlet pressure (respectively abs(INTIAL) == 2
    # and abs(INTIAL) == 3) to take into account that channels may have \
    # different values of these pressures and that it will equalize instantly.

    # Evaluate average known pressure (cdp, 09/2020)
    p_ave = np.sum(p_known) / N_group
    # Initialization
    rho = np.zeros(N_group)
    mu = np.zeros(N_group)
    Re = np.zeros(N_group)
    fric = np.zeros(N_group)
    flow_dir = list()
    for ii in range(N_group):
        fluid_comp = chan_group[ii]
        flow_dir.append(fluid_comp.coolant.operations.flow_direction)
        # Invoke method eval_coolant_density_din_viscosity_gen_flow to evaluate density and dynamic viscosity at average known pressure and inlet temperature
        (
            rho[ii],
            mu[ii],
        ) = fluid_comp.coolant.eval_coolant_density_din_viscosity_gen_flow(
            [p_ave], [T_inl[ii]]
        )
        # Invoke method eval_reynolds_from_mass_flow_rate to evaluate Reynolds number at average known pressure and inlet temperature
        Re[ii] = fluid_comp.coolant.eval_reynolds_from_mass_flow_rate(
            mdot_known[ii], mu[ii]
        )
        # Friction factor at average known pressure and inlet temperature, nodal = None specities that total friction factor in Gen_Flow module is evaluated (cdp, 09/2020)
        fluid_comp.channel.eval_friction_factor(Re[ii], nodal=None)
        fric[ii] = fluid_comp.channel.friction_factors[None].total
    # End for ii.
    # Pressure drop evaluated with known properties, to be able to deal with \
    # any fluid type; array smart notation (cdp, 09/2020)
    # Evaluate coefficient of the therm of second degree (cdp, 09/2020)
    alpha = g0 * fric / rho
    # Compute total group mass flow rate (cdp, 09/2020)
    mdot_known_group = np.sum(mdot_known)
    # Evaluate channel group pressure drop (cdp, 09/2020)
    delta_p_group = (abs(mdot_known_group) / np.sum(1 / np.sqrt(alpha))) ** 2
    if flow_dir.count(FlowDirection.FORWARD) == len(flow_dir):
        print("Forward flow\n")
    elif flow_dir.count(FlowDirection.BACKWARD) == len(flow_dir):
        print("Backward flow\n")
    else:
        raise ValueError(
            "Channels in hydraulic parallel must all have the same flow direction.\nPlease, check flag FLOWDIR in sheet CHANNEL of input file conductor_operation.xlsx.\n"
        )
    # End if flow_dir.count("forward") != len(flow_dir)
    # Real mass flow rate distribution, absolute value
    mdot_known = np.sqrt(delta_p_group * rho / (g0 * fric))
    # Evaluate relative error on mass flow rate (cdp, 09/2020)
    error_mfr = abs((np.sum(mdot_known) - mdot_known_group) / mdot_known_group)
    if error_mfr > tol:
        warnings.warn(
            f"""Channel in hydraulic parallel:\nabs(INTIAL) = {abs(intial)}\nerror_mfr = {error_mfr}\nis larger than specified tolerance ({tol}); this may have some effects in the solution.\n"""
        )
    # Evaluate missing pressure (cdp, 09/2020)
    if abs(intial) == 2:
        # Outlet pressure (cdp, 09/2020)
        p_missing = p_ave - delta_p_group
    elif abs(intial) == 3:
        # Inlet pressure (cdp, 09/2020)
        p_missing = p_ave + delta_p_group
    # Loop on channes (cdp, 09/2020)
    for ii in range(N_group):
        fluid_comp = chan_group[ii]
        if p_missing != getattr(fluid_comp.coolant.operations, key_b):
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated {word} pressure is 
        different from the one in Worksheet CHAN of input file 
        {cond.file_paths.operation_path}:
        {p_missing} != {getattr(fluid_comp.coolant.operations, key_b)}.\n
        This value is overwritten by the evaluated one:\n
        {symbol} = {p_missing} Pa\n"""
            )
            # overwriting channel missing pressure (cdp, 09/2020)
            setattr(fluid_comp.coolant.operations, key_b, p_missing)
        if mdot_known[ii] != getattr(fluid_comp.coolant.operations, key_mfr):
            warnings.warn(
                f"""Function {gen_flow.__name__}, {fluid_comp.identifier}, INTIAL == 
        {fluid_comp.coolant.operations.hydraulic_bc_type.name}. Evaluated mass flow rate is 
        different from the one in Worksheet CHAN of input file 
        {cond.file_paths.operation_path}:
        {mdot_known[ii]} != {getattr(fluid_comp.coolant.operations, key_mfr)}.\n
        This value is overwritten by the evaluated one:\n
        mdot_known = {mdot_known[ii]} Pa\n"""
            )
            # overwriting channel inlet mass flow rate and assign the correct sign according to the flow direction.
            setattr(
                fluid_comp.coolant.operations,
                key_mfr,
                fluid_comp.channel.flow_sign * mdot_known[ii],
            )


# end function Abs_INTIAL_equal_2_or_5_hp (cdp, 09/2020)




def get_inlet_conductor_mfr(cond):

    """
    Function that evaluates total conductor inlet mass flow rate (MDTINL) as the sum of channels inlet mass flow rates together with channels flow fraction (flow_fraction). This function must be invoked only after that inlet mass flow rate is evaluated for each channel. (cdp, 09/2020)
    """

    # total conductor inlet mass flow rate initialization (cdp, 09/2020)
    cond.MDTINL = 0.0
    # Loop on FluidComponent to compute conductor inlet mass flow rate \
    # (cdp, 09/2020)
    for fluid_comp in cond.inventory.fluids.collection:
        cond.MDTINL = cond.MDTINL + fluid_comp.coolant.operations.inlet_mass_rate
    # Loop on FluidComponent to compute channels flow fraction (cdp, 09/2020)
    for fluid_comp in cond.inventory.fluids.collection:
        if cond.MDTINL != 0.0:
            # Avoid division by 0 if INTIAL = 3 or INTIAL = 4
            fluid_comp.channel.flow_fraction = (
                fluid_comp.coolant.operations.inlet_mass_rate / cond.MDTINL
            )
        else:
            fluid_comp.channel.flow_fraction = 0.0


# end function Get_inlet_conductor_mfr
