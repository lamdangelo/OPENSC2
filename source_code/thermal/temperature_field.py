"""
This module owns temperature-field operations for the thermal problem:
initialization of the solid-component temperature spatial distribution, and
Gauss-point temperature evaluation.

Relocated from ``utility_functions/solid_components_initialization.py`` and
``conductor/conductor.py`` as part of consolidating the thermal calculations
into the ``thermal`` package.
"""

import numpy as np

from physical_fields.physical_field import FieldContainer, GridLocation


def eval_temperature_solids_gauss_point(conductor: object) -> None:
    """Evaluate temperature of SolidComponents in Gauss points."""

    # Loop on SolidComponent
    for obj in conductor.inventory.solids.collection:
        obj.gauss_fields.temperature = (
            np.abs(
                obj.node_fields.temperature[:-1]
                + obj.node_fields.temperature[1:]
            )
            / 2.0
        )


def solid_components_temperature_initialization(cond):
    """
    Function that initializes Solid Components temperature spatial distribution
    according to conductor topology and to the value of flag INTIAL
    (``operations.initial_temperature_mode``).
    """
    T_min = np.zeros(cond.inventory.fluids.number)
    # Loop on FluidComponent to get the minimum temperature among all the \
    # channels (cdp, 12/2020)
    for rr, fluid_comp in enumerate(cond.inventory.fluids.collection):
        T_min[rr] = fluid_comp.coolant.node_fields.temperature.min()
    # end for rr (cdp, 12/2020)
    # For each solid component evaluate temperature (cdp, 07/2020)
    # If needed read only the sub matrix describing channel - solid objects \
    # contact (cdp, 07/2020)
    # nested loop on channel - solid objects (cpd 07/2020)
    for cc, s_comp in enumerate(cond.inventory.solids.collection):
        s_comp.node_fields = FieldContainer(GridLocation.NODE)
        s_comp.gauss_fields = FieldContainer(GridLocation.GAUSS)
        # s_comp temperature initialization to 0 (cdp, 12/2020)
        s_comp.node_fields.temperature = np.zeros(cond.mesh.number_of_nodes)
        if s_comp.operations.initial_temperature_mode == 0:
            # Not user defined temperature initialization (cdp, 12/2020)
            # Read the channel - solid column of the contact perimeter matrix
            # to determine whether s_comp is in thermal contact with channels.
            # Row/column 0 of the coupling matrices is the Environment, hence
            # the offset of 1.
            weight = cond.coupling.contact_perimeter.matrix[
                1 : 1 + cond.inventory.fluids.number,
                1 + cond.inventory.fluids.number + cc,
            ]
            if np.sum(weight) > 0:
                # evaluate SolidComponent temperature as the weighted average on \
                # conctat_perimeter with channels (cpd 07/2020)
                for rr, fluid_comp in enumerate(
                    cond.inventory.fluids.collection
                ):
                    s_comp.node_fields.temperature = s_comp.node_fields.temperature + fluid_comp.coolant.node_fields.temperature * weight[
                        rr
                    ] / np.sum(
                        weight
                    )
            else:
                # The s_comp object is not in thermal contact with channels: the \
                # temperature spatial distribution is initialized at the minimum \
                # temperature among the channels (cdp, 12/2020)
                s_comp.node_fields.temperature = (
                    np.ones(cond.mesh.number_of_nodes) * T_min.min()
                )
            # end if np.sum(weight) (cdp, 12/2020)
        elif abs(s_comp.operations.initial_temperature_mode) == 1:
            # User imposed initial temperature spatial distribution (cdp, 12/2020)
            if s_comp.operations.initial_temperature_mode == 1:
                # linear spatial temperature distribution (cdp, 12/2020)
                s_comp.node_fields.temperature = np.interp(
                    cond.mesh.node_coordinates,
                    [0.0, cond.inputs.zlength],
                    [s_comp.operations.inlet_temperature,
                     s_comp.operations.outlet_temperature],
                )

            elif s_comp.operations.initial_temperature_mode == -1:
                print("still to do\n")
        else:
            # raise error due to not available INTIAL value (cdp, 12/2020)
            raise ValueError(
                f"""INTIAL value not available. Please check check INTIAL value in sheet {s_comp.name} of file {cond.file_paths.operation_path}.\n"""
            )
        # end if s_comp.operations.initial_temperature_mode (cdp, 12/2020)
    # end for cc (cdp, 12/2020)
