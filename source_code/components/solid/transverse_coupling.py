"""Nonlocal transverse insulation coupling between solid components.

File-driven heat exchange between winding-geometry-adjacent positions
(turn-to-turn and layer-to-layer contact through the insulation) that the
1D metric cannot see. The patch rows of a component (read through
ports.external_inputs at construction) are mapped onto its mesh here, once
every conductor exists; SolidComponent.get_transverse_coupling evaluates the
exchange at every step.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def partner_index(conductors: Iterable) -> dict:
    """{(conductor identifier, component identifier): {"component",
    "coordinates"}} over the strands and jackets of all conductors."""
    partners = {}
    for conductor in conductors:
        for collection in (
            conductor.inventory.strands.collection,
            conductor.inventory.jackets.collection,
        ):
            for component in collection:
                partners[(conductor.identifier, component.identifier)] = {
                    "component": component,
                    "coordinates": conductor.mesh.node_coordinates,
                }
    return partners


def build_transverse_patches(
    rows: Iterable, coordinates: np.ndarray, partners: Mapping, owner: str
) -> list:
    """Map the patch rows of one component onto its node coordinates.

    Each patch dict holds the local node ``indices`` in [x_low, x_high], the
    partner positions ``partner_x`` they map onto linearly (a decreasing
    partner interval encodes reversed orientation), the conductance and its
    temperature-dependence parameters, the ``partner`` entry of
    :func:`partner_index` (None for a BOUNDARY reservoir) and the
    ``boundary_temperature`` (None unless BOUNDARY). Rows that cover no
    local node are dropped.

    Raises:
        KeyError: if a row names a partner that no conductor provides.
    """
    patches = []
    for row in rows:
        if row.is_boundary:
            partner = None
            boundary_temperature = row.boundary_temperature
        else:
            partner_key = (row.partner_conductor, row.partner_component)
            if partner_key not in partners:
                raise KeyError(
                    f"transverse coupling of {owner}: unknown "
                    f"partner {partner_key[0]}/{partner_key[1]}"
                )
            partner = partners[partner_key]
            boundary_temperature = None
        mask = (coordinates >= row.x_low) & (coordinates <= row.x_high)
        if not mask.any():
            continue
        fraction = (coordinates[mask] - row.x_low) / (row.x_high - row.x_low)
        patches.append(
            {
                "indices": np.flatnonzero(mask),
                "partner_x": row.partner_x_start
                + fraction * (row.partner_x_end - row.partner_x_start),
                "conductance": row.conductance,
                "temperature_exponent": row.temperature_exponent,
                "reference_temperature": row.reference_temperature,
                "saturation_temperature": row.saturation_temperature,
                "partner": partner,
                "boundary_temperature": boundary_temperature,
            }
        )
    return patches
