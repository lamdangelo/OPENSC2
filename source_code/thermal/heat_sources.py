"""
This module owns the assembly of the heat-source terms (external heating,
Joule heating, radiative exchange) at nodal and Gauss points, for the
strand and jacket components of a conductor.

Relocated from ``conductor/conductor.py`` as part of consolidating the
thermal calculations into the ``thermal`` package.
"""

from conductor.conductor_flags import HTC_Choice
from electromagnetics.resistance import (
    distribute_joule_power_to_parallel_jackets,
)
from thermal.thermal_flags import HeatExcitation

_RADIATIVE_HTC_CHOICES = (
    HTC_Choice.RADIATIVE_HTC_COMPUTED,
    HTC_Choice.RADIATIVE_HTC_READ_FROM_FILE,
)


def build_heat_source(conductor: object, simulation: object) -> None:
    """Build heat source therms in nodal and Gauss points for strand and jacket objects.

    Args:
        conductor (object): ConductorComponent object with all information needed for the computation.
        simulation (object): object with all information about the simulation.
    """
    # Refactor both private methods!
    _build_heat_source_nodal_pt(conductor, simulation)
    _build_heat_source_gauss_pt(conductor)


def _build_heat_source_nodal_pt(conductor: object, simulation: object) -> None:
    """Build heat source therms in nodal points for strand and jacket objects.

    Args:
        conductor (object): ConductorComponent object with all information needed for the computation.
        simulation (object): object with all information about the simulation.
    """

    # Alias
    interf_flag = conductor.coupling.contact_perimeter_flag

    # Loop on StrandComponent objects.
    for strand in conductor.inventory.strands.collection:
        if conductor.cond_num_step == 0 and strand.operations.heat_flux_mode == HeatExcitation.NO_HEATING:
            # Call method get_heat only once to initialize key EXTFLX of
            # dictionary node_fields to zeros.
            strand.get_heat(conductor)
        elif strand.operations.heat_flux_mode != HeatExcitation.NO_HEATING:
            # Call method get_heat to evaluate external heating only if
            # heating is on.
            strand.get_heat(conductor)

        # Call method jhtflx_new_0 to initialize JHTFLX to zeros for each
        # conductor solid components.
        strand.jhtflx_new_0(conductor)
        # Evaluate joule power due to electric resistance along strand
        # object.
        strand.get_joule_power_along(conductor)
        # Evaluate joule power due to electric conductance across strand
        # object.
        strand.get_joule_power_across(conductor)
        # Call set_energy_counters to initialize EEXT and EJHT to zeros for
        # each conductor solid components.
        strand.set_energy_counters(conductor)

    # Loop on JacketComponents objects.
    for rr, jacket in enumerate(conductor.inventory.jackets.collection):
        if conductor.cond_num_step == 0 and jacket.operations.heat_flux_mode == HeatExcitation.NO_HEATING:
            # Call method get_heat only once to initialize key EXTFLX of
            # dictionary node_fields to zeros.
            jacket.get_heat(conductor)
        elif jacket.operations.heat_flux_mode != HeatExcitation.NO_HEATING:
            # Call method get_heat to evaluate external heating only if
            # heating is on.
            jacket.get_heat(conductor)

        # Call method jhtflx_new_0 to initialize JHTFLX to zeros for each
        # conductor solid components.
        jacket.jhtflx_new_0(conductor)
        # Evaluate joule power due to electric resistance along jacket
        # object.
        jacket.get_joule_power_along(conductor)
        # Evaluate joule power due to electric conductance across jacket
        # object.
        jacket.get_joule_power_across(conductor)
        # Call set_energy_counters to initialize EEXT and EJHT to zeros for
        # each conductor solid components.
        jacket.set_energy_counters(conductor)
        if (
            abs(interf_flag[
                simulation.environment.KIND, jacket.identifier
            ]) == 1
        ):
            # Evaluate the external heat by radiation in nodal points.
            jacket._radiative_source_therm_env(conductor, simulation.environment)
        # End if abb(interf_flag)
        for _, jacket_c in enumerate(
            conductor.inventory.jackets.collection[rr + 1 :]
        ):
            if (
                HTC_Choice.get_htc_choice_flag(
                    conductor.coupling.htc_choice[
                        jacket.identifier, jacket_c.identifier
                    ]
                )
                in _RADIATIVE_HTC_CHOICES
            ):
                # Evaluate the inner heat exchange by radiation in nodal
                # points.
                jacket._radiative_heat_exc_inner(conductor, jacket_c)
                jacket_c._radiative_heat_exc_inner(conductor, jacket)
            # End if abs.
        # End for jacket_c.
    # End for rr.

    # Move each ideal-parallel jacket's Joule share from its strand's power
    # array to the jacket's (current redistribution on quench).
    distribute_joule_power_to_parallel_jackets(conductor)


def _build_heat_source_gauss_pt(conductor: object) -> None:
    """Build heat source therms in Gauss points for strand and jacket objects."""

    # Loop on StrandComponent objects.
    for strand in conductor.inventory.strands.collection:

        strand.gauss_fields.Q1 = (
            strand.node_fields.JHTFLX[:-1]
            + strand.node_fields.EXTFLX[:-1]
            + strand.node_fields.total_linear_power_el_cond[:-1]
            + strand.gauss_fields.linear_power_el_resistance
        )

        strand.gauss_fields.Q2 = (
            strand.node_fields.JHTFLX[1:]
            + strand.node_fields.EXTFLX[1:]
            + strand.node_fields.total_linear_power_el_cond[1:]
            + strand.gauss_fields.linear_power_el_resistance
        )

    # Loop on JacketComponents objects.
    for rr, jacket in enumerate(conductor.inventory.jackets.collection):

        jacket.gauss_fields.Q1 = (
            jacket.node_fields.JHTFLX[:-1]
            + jacket.node_fields.EXTFLX[:-1]
            + jacket.gauss_fields.linear_power_el_resistance
        )
        jacket.gauss_fields.Q2 = (
            jacket.node_fields.JHTFLX[1:]
            + jacket.node_fields.EXTFLX[1:]
            + jacket.gauss_fields.linear_power_el_resistance
        )
        # Add the radiative heat contribution with the environment.
        jacket.gauss_fields.Q1 = (
            jacket.gauss_fields.Q1 + jacket.radiative_heat_env[:-1]
        )
        jacket.gauss_fields.Q2 = (
            jacket.gauss_fields.Q2 + jacket.radiative_heat_env[1:]
        )

    # Separate nested loop is needed in order to define quantities
    # gauss_fields.Q1 and gauss_fields.Q2 for component jacket_c in
    # the previous loop, otherwise a key error will be raised.
    for rr, jacket in enumerate(conductor.inventory.jackets.collection):
        # Nested loop jacket - jacket.
        for _, jacket_c in enumerate(
            conductor.inventory.jackets.collection[rr + 1 :]
        ):
            key = f"{jacket.identifier}_{jacket_c.identifier}"
            # Add the radiative heat contribution between inner surface of the enclosure and inner jackets.
            jacket.gauss_fields.Q1 = (
                jacket.gauss_fields.Q1 + jacket.radiative_heat_inn[key][:-1]
            )
            jacket.gauss_fields.Q2 = (
                jacket.gauss_fields.Q2 + jacket.radiative_heat_inn[key][1:]
            )

            jacket_c.gauss_fields.Q1 = (
                jacket_c.gauss_fields.Q1 + jacket_c.radiative_heat_inn[key][:-1]
            )
            jacket_c.gauss_fields.Q2 = (
                jacket_c.gauss_fields.Q2 + jacket_c.radiative_heat_inn[key][1:]
            )
        # End for jacket_c.
    # end for jacket.
