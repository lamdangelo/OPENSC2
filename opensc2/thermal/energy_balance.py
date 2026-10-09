"""
This module owns end-of-simulation energy and mass conservation diagnostics:
the final integrated thermal energy stored in the solid components, and the
mass/energy balance between fluid inlet/outlet and environment heat exchange.

Relocated from ``conductor/conductor.py`` (the body of ``post_processing`` and
the whole of ``mass_energy_balance``) as part of splitting the file for
single responsibility.
"""

import numpy as np


def evaluate_final_energy(conductor: object) -> None:
    """Evaluate the total final thermal energy stored in the solid components.

    Used to check the imposition of SolidComponent temperature initial
    spatial distribution. Updates ``conductor.balances.E_sol_fin``,
    ``conductor.balances.E_str_fin`` (strands) and ``conductor.balances.E_jk_fin`` (jackets).
    """
    for s_comp in conductor.inventory.solids.collection:
        conductor.balances.E_sol_fin = conductor.balances.E_sol_fin + s_comp.inputs.cross_section * np.sum(
            (
                conductor.mesh.node_coordinates[1 : conductor.mesh.number_of_nodes]
                - conductor.mesh.node_coordinates[0:-1]
            )
            * s_comp.gauss_fields.total_density
            * s_comp.gauss_fields.total_isobaric_specific_heat
            * s_comp.gauss_fields.temperature
        )
        if s_comp.name != "Z_JACKET":
            conductor.balances.E_str_fin = conductor.balances.E_str_fin + s_comp.inputs.cross_section * np.sum(
                (
                    conductor.mesh.node_coordinates[1 : conductor.mesh.number_of_nodes]
                    - conductor.mesh.node_coordinates[0:-1]
                )
                * s_comp.gauss_fields.total_density
                * s_comp.gauss_fields.total_isobaric_specific_heat
                * s_comp.gauss_fields.temperature
            )
        else:
            conductor.balances.E_jk_fin = conductor.balances.E_jk_fin + s_comp.inputs.cross_section * np.sum(
                (
                    conductor.mesh.node_coordinates[1 : conductor.mesh.number_of_nodes]
                    - conductor.mesh.node_coordinates[0:-1]
                )
                * s_comp.gauss_fields.total_density
                * s_comp.gauss_fields.total_isobaric_specific_heat
                * s_comp.gauss_fields.temperature
            )
    # end for s_comp (cdp, 12/2020)


def evaluate_mass_energy_balance(conductor: object, simulation: object) -> None:
    """Evaluate mass and energy balance on the conductor.

    Updates ``conductor.balances.mass_balance``, ``conductor.balances.energy_balance``,
    ``conductor.balances.inner_pow`` and ``conductor.balances.outer_pow``.
    """

    # Alias
    interf_flag = conductor.coupling.contact_perimeter_flag

    conductor.balances.mass_balance = 0.0  # mass balance initialization (cdp, 09/2020)
    conductor.balances.energy_balance = 0.0  # energy balance initialization (cdp, 09/2020)
    conductor.balances.inner_pow = 0.0
    conductor.balances.outer_pow = 0.0
    for fluid_comp in conductor.inventory.fluids.collection:
        # Mass balance (cdp, 09/2020)
        conductor.balances.mass_balance = conductor.balances.mass_balance + conductor.time_state.time_step * (
            fluid_comp.coolant.node_fields.mass_flow_rate[0]
            - fluid_comp.coolant.node_fields.mass_flow_rate[-1]
        )
        # Energy balance: sum(mdot_inl*(w_inl + v_inl^2/2) - \
        # mdot_out*(w_out + v_out^2/2)) (cdp, 09/2020)
        conductor.balances.energy_balance = conductor.balances.energy_balance + conductor.time_state.time_step * (
            fluid_comp.coolant.node_fields.mass_flow_rate[0]
            * (
                fluid_comp.coolant.node_fields.total_enthalpy[0]
                + fluid_comp.coolant.node_fields.velocity[0] ** 2 / 2.0
            )
            - fluid_comp.coolant.node_fields.mass_flow_rate[-1]
            * (
                fluid_comp.coolant.node_fields.total_enthalpy[-1]
                + fluid_comp.coolant.node_fields.velocity[-1] ** 2 / 2.0
            )
        )
        conductor.balances.inner_pow = conductor.balances.inner_pow + fluid_comp.coolant.node_fields.mass_flow_rate[0] * (
            fluid_comp.coolant.node_fields.total_enthalpy[0]
            + fluid_comp.coolant.node_fields.velocity[0] ** 2 / 2.0
        )
        conductor.balances.outer_pow = conductor.balances.outer_pow + fluid_comp.coolant.node_fields.mass_flow_rate[-1] * (
            fluid_comp.coolant.node_fields.total_enthalpy[-1]
            + fluid_comp.coolant.node_fields.velocity[-1] ** 2 / 2.0
        )
    # End for fluid_comp.
    for jacket in conductor.inventory.jackets.collection:
        if (
            abs(interf_flag[
                    simulation.environment.KIND, jacket.identifier
                ]
            ) == 1
        ):
            key = f"{simulation.environment.KIND}_{jacket.identifier}"
            conductor.balances.energy_balance = (
                conductor.balances.energy_balance
                + conductor.time_state.time_step
                * jacket.inputs.outer_perimeter
                * np.sum(
                    (
                        conductor.gauss_fields.HTC["env_sol"][key]["conv"]
                        + conductor.gauss_fields.HTC["env_sol"][key]["rad"]
                    )
                    * conductor.mesh.element_lengths
                    * (
                        simulation.environment.temperature
                        - jacket.gauss_fields.temperature
                    )
                )
            )
            conductor.balances.inner_pow = conductor.balances.inner_pow + jacket.inputs.outer_perimeter * np.sum(
                (
                    conductor.gauss_fields.HTC["env_sol"][key]["conv"]
                    + conductor.gauss_fields.HTC["env_sol"][key]["rad"]
                )
                * conductor.mesh.element_lengths
                * (
                    simulation.environment.temperature
                    - jacket.gauss_fields.temperature
                )
            )
        # End if.
    # End jacket.
    # Energy balance to check the correct management of SolidComponent forced \
    # initial temperature distribution (cdp, 12/2020)
    # E_residual = (conductor.balances.E_sol_ini - conductor.balances.E_sol_fin) - \
    #              (conductor.balances.enthalpy_inl - conductor.balances.enthalpy_out)
    # print(f"E_sol_ini = {conductor.balances.E_sol_ini} J\n")
    # print(f"E_sol_fin = {conductor.balances.E_sol_fin} J\n")
    # print(f"E_str_ini = {conductor.balances.E_str_ini} J\n")
    # print(f"E_str_fin = {conductor.balances.E_str_fin} J\n")
    # print(f"E_jk_ini = {conductor.balances.E_jk_ini} J\n")
    # print(f"E_jk_fin = {conductor.balances.E_jk_fin} J\n")
    # print(f"enthalpy_inl = {conductor.balances.enthalpy_inl} J\n")
    # print(f"enthalpy_out = {conductor.balances.enthalpy_out} J\n")
    # print(f"enthalpy_balance = {conductor.balances.enthalpy_balance} J\n")
    # print(f"E_residual = {E_residual} J\n")

    print(f"Energy balance = {conductor.balances.energy_balance} J\n")
    # Print outer inner power ration only if inner_pow is not 0.0
    if conductor.balances.inner_pow != 0.0:
        print(f"Outer inner power ratio % = {1e2*conductor.balances.outer_pow/conductor.balances.inner_pow} ~")
