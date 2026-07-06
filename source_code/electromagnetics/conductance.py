"""
This module evaluates the transverse electric conductance between
current-carrying strand components that are in contact, and assembles the
resulting sparse conductance matrix.

Two conductance modes are supported (see ``ElectricConductanceMode``):

* ``CONDUCTANCE_PER_UNIT_LENGTH``:
    ``G = sigma * gauss_node_distance``
    where ``sigma`` is the conductance per unit length [S/m].

* ``ACTUAL_CONDUCTANCE_BETWEEN_COMPONENTS``:
    ``G = sigma * contact_perimeter * avg_gauss_node_distance / transverse_distance``
    where ``sigma`` is the actual conductance [S], i.e. the value already
    accounts for the contact geometry.

The ``electric_conductance_mode`` matrix on the conductor stores raw integer
values read from the Excel input (0 = none, 1 = per unit length, 2 = actual).
The ``ElectricConductanceMode.get_electric_conductance_mode`` translator is
applied element-wise before dispatching to the appropriate formula.

Ported from the private methods of the Conductor class:
    __evaluate_transversal_distance
    __evaluate_electric_conductance
    __build_electric_conductance_matrix
"""

import numpy as np
from scipy.sparse import diags

from electromagnetics.electromagnetic_flags import ElectricConductanceMode


def evaluate_transversal_distance(conductor: object) -> np.ndarray:
    """Compute the transverse distance between each pair of contacting nodes.

    For every row in ``conductor.contact_nodes_current_carriers`` the
    Euclidean distance between the two strand nodes (in the x-y plane and
    including z, i.e. the full 3-D distance) is evaluated from the nodal
    coordinates.

    Args:
        conductor: Conductor object with ``nodal_coordinates`` and
            ``contact_nodes_current_carriers`` set up.

    Returns:
        Array of distances, one per contact pair row, shape
        ``(n_contact_pairs,)``.
    """
    contact_nodes = conductor.contact_nodes_current_carriers
    strand_coords = conductor.nodal_coordinates.loc["StrandComponent"]

    distance = (
        (
            (
                strand_coords
                .iloc[contact_nodes["end"], :]
                .reset_index(level="Identifier", drop=True)
                - strand_coords
                .iloc[contact_nodes["start"], :]
                .reset_index(level="Identifier", drop=True)
            )
            ** 2
        )
        .sum(axis="columns")
        .apply(np.sqrt)
        .to_numpy()
    )
    return distance


def evaluate_electric_conductance(
    conductor: object, distance: np.ndarray
) -> np.ndarray:
    """Evaluate the electric conductance for each transverse contact element.

    Iterates over the contact pairs on the first cross-section and dispatches
    to the per-unit-length or actual-conductance formula based on the mode
    stored in ``conductor.electric_conductance_mode``.

    Args:
        conductor: Conductor object with ``electric_conductance``,
            ``electric_conductance_mode``, ``gauss_node_distance``,
            ``coupling``, and ``_contact_nodes_first`` set up.
        distance: Transverse distances between contacting node pairs as
            returned by :func:`evaluate_transversal_distance`.

    Returns:
        Array of conductance values, one per row in
        ``conductor.contact_nodes_current_carriers``.
    """
    n_contacts_first = conductor._contact_nodes_first.shape[0]
    electric_conductance = np.zeros(
        conductor.contact_nodes_current_carriers.shape[0]
    )

    n_fluid = conductor.inventory.fluids.number
    n_all = conductor.inventory.all_components.number

    for ii, indexes in enumerate(conductor._contact_nodes_first):
        # Matrix indices into the coupling data arrays (offset for fluids + env)
        mat_ind = indexes + n_fluid + 1
        # Array indices into gauss_node_distance (offset for fluids only)
        arr_ind = indexes + n_fluid

        raw_mode = int(conductor.electric_conductance_mode[indexes[0], indexes[1]])
        mode = ElectricConductanceMode.get_electric_conductance_mode(raw_mode)

        sigma = conductor.electric_conductance[indexes[0], indexes[1]]
        gnd_i = conductor.gauss_node_distance[arr_ind[0]::n_all]
        gnd_j = conductor.gauss_node_distance[arr_ind[1]::n_all]
        dist_slice = distance[ii::n_contacts_first]

        if mode == ElectricConductanceMode.CONDUCTANCE_PER_UNIT_LENGTH:
            # G = sigma * gauss_node_distance
            electric_conductance[ii::n_contacts_first] = sigma * gnd_i

        elif mode == ElectricConductanceMode.ACTUAL_CONDUCTANCE_BETWEEN_COMPONENTS:
            # G = sigma * contact_perimeter * avg_gauss_node_distance / transverse_distance
            contact_perimeter = conductor.coupling.contact_perimeter[mat_ind[0], mat_ind[1]]
            electric_conductance[ii::n_contacts_first] = (
                sigma * contact_perimeter * (gnd_i + gnd_j) / 2.0 / dist_slice
            )
        # ElectricConductanceMode.NO_CONDUCTANCE leaves the slice as zero

    return electric_conductance


def build_conductance_matrix(conductor: object) -> None:
    """Build the sparse electric conductance matrix for transverse contacts.

    Evaluates the transversal distances, computes per-contact conductances,
    and assembles the node-level conductance matrix via:

        ``G_node = B^T * diag(G) * B``

    where ``B`` is the contact incidence matrix.

    Results stored in:
    * ``conductor.electric_conductance_diag_matrix``: diagonal matrix of
      per-contact conductances.
    * ``conductor.electric_conductance_matrix``: the assembled node-level
      conductance matrix.

    Args:
        conductor: Conductor object with all contact and conductance
            data already set up.
    """
    distance = evaluate_transversal_distance(conductor)
    electric_conductance = evaluate_electric_conductance(conductor, distance)

    n_contacts = conductor.contact_nodes_current_carriers.shape[0]
    conductor.electric_conductance_diag_matrix = diags(
        electric_conductance,
        offsets=0,
        shape=(n_contacts, n_contacts),
        format="csr",
        dtype=float,
    )

    conductor.electric_conductance_matrix = (
        conductor.contact_incidence_matrix.T
        @ conductor.electric_conductance_diag_matrix
        @ conductor.contact_incidence_matrix
    )
