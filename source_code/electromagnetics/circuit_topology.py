"""
This module builds the circuit-level topology data structures for the
electric problem: element-to-node connectivity, longitudinal incidence
matrices, and transverse contact incidence matrices — all limited to the
current-carrying strand components.

Ported from the private methods of the Conductor class:
    __build_connectivity_current_carriers
    __build_incidence_matrix
    __contact_current_carriers_first_cross_section
    __contact_current_carriers
    __build_contact_incidence_matrix
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def build_connectivity_current_carriers(conductor: object) -> None:
    """Fill the connectivity matrix for current-carrying strand components.

    Each strand occupies every n-th row of the connectivity matrix, where
    n is the total number of strands. Start and end node indices are computed
    by distributing nodes across strands and mesh elements.

    Results are stored in ``conductor.connectivity_matrix_current_carriers``.

    Args:
        conductor: Conductor object whose connectivity matrix, inventory, and
            mesh are already initialised.
    """
    n_strands = conductor.inventory.strands.number
    n_elements = conductor.mesh.number_of_elements
    n_elements_total = conductor.total_elements_current_carriers

    for ii, strand in enumerate(conductor.inventory.strands.collection):
        nodes = np.linspace(ii, ii + n_elements_total, n_elements + 1, dtype=int)
        cm = conductor.connectivity_matrix_current_carriers
        cm.iloc[ii::n_strands, cm.columns.get_loc("start")] = nodes[:-1]
        cm.iloc[ii::n_strands, cm.columns.get_loc("end")] = nodes[1:]
        cm.iloc[ii::n_strands, cm.columns.get_loc("identifiers")] = strand.identifier


def build_incidence_matrix(conductor: object) -> None:
    """Build the edge-to-node incidence matrix for current-carrying components.

    The incidence matrix ``A`` has shape ``(n_elements, n_nodes)`` with entries
    ``-1`` at the start node and ``+1`` at the end node of every element.
    The transposed form is stored alongside it.

    Results are stored in ``conductor.incidence_matrix`` and
    ``conductor.incidence_matrix_transposed``.

    Reference: formulation from professor F. Freschi (Politecnico di Torino).

    Args:
        conductor: Conductor object with a filled
            ``connectivity_matrix_current_carriers``.
    """
    n_elements = conductor.total_elements_current_carriers
    n_nodes = conductor.total_nodes_current_carriers

    # Row indices: each element appears twice (start and end node)
    irow = np.tile(np.r_[0:n_elements], (2, 1)).flatten("F")

    # Column indices: start and end node for every element
    jcol = (
        conductor.connectivity_matrix_current_carriers.iloc[:, 0:2]
        .to_numpy()
        .copy()
        .transpose()
        .flatten("F")
    )

    # -1 at start node, +1 at end node
    values = np.tile([-1, 1], n_elements)

    conductor.incidence_matrix = coo_matrix(
        (values, (irow, jcol)), shape=(n_elements, n_nodes)
    ).tocsr()
    conductor.incidence_matrix_transposed = conductor.incidence_matrix.T


def _detect_contacts_first_cross_section(conductor: object) -> None:
    """Identify pairs of strands in contact on the first cross-section.

    Reads the ``contact_perimeter_flag`` coupling matrix to find which
    strand pairs have a non-zero contact flag. Results (one row per contact
    pair) are stored in the private attribute
    ``conductor._contact_nodes_first`` as an integer array of shape
    ``(n_contacts, 2)``.

    Args:
        conductor: Conductor object with ``coupling`` and
            ``inventory`` already set up.
    """
    interf_flag = conductor.coupling.contact_perimeter_flag
    base = 1 + conductor.inventory.fluids.number
    limit = base + conductor.inventory.strands.number

    rows = []
    for row in range(base, limit):
        contacts = np.nonzero(
            np.abs(interf_flag.matrix[row, base:limit]) == 1
        )[0]
        if contacts.size > 0:
            rows.append(
                np.column_stack(
                    (np.full(contacts.size, row - base, dtype=int), contacts)
                )
            )

    conductor._contact_nodes_first = (
        np.vstack(rows) if rows else np.empty((0, 2), dtype=int)
    )


def detect_contacts(conductor: object) -> None:
    """Detect all contact node pairs between current-carrying strands.

    Calls :func:`_detect_contacts_first_cross_section` to obtain contacts on
    the first cross-section, then replicates them across every cross-section
    by adding the appropriate node offsets.

    Results are stored in ``conductor.contact_nodes_current_carriers`` as a
    DataFrame with columns ``"start"`` and ``"end"``.

    Args:
        conductor: Conductor object with inventory and mesh set up.
    """
    _detect_contacts_first_cross_section(conductor)

    if conductor._contact_nodes_first.size == 0:
        conductor.contact_nodes_current_carriers = pd.DataFrame(
            columns=["start", "end"], dtype=int
        )
        return

    n_cross_sections = conductor.mesh.number_of_elements + 1
    n_strands = conductor.inventory.strands.number

    offsets = np.arange(n_cross_sections, dtype=int)[:, None] * n_strands
    contact_nodes = (
        conductor._contact_nodes_first[np.newaxis, :, :] + offsets[:, None]
    ).reshape(-1, 2)

    conductor.contact_nodes_current_carriers = pd.DataFrame(
        contact_nodes, dtype=int, columns=["start", "end"]
    )


def build_contact_incidence_matrix(conductor: object) -> None:
    """Build the edge-to-node incidence matrix for transverse strand contacts.

    Constructs a sparse ``(n_contacts, n_nodes)`` matrix with ``-1`` at the
    start node and ``+1`` at the end node for each transverse contact element.

    Result is stored in ``conductor.contact_incidence_matrix``.

    Args:
        conductor: Conductor object with
            ``contact_nodes_current_carriers`` already built.
    """
    n_contacts = conductor.contact_nodes_current_carriers.shape[0]

    row_ind = np.tile(np.r_[0:n_contacts], (2, 1)).flatten("F")
    col_ind = (
        conductor.contact_nodes_current_carriers.to_numpy().transpose().flatten("F")
    )

    conductor.contact_incidence_matrix = coo_matrix(
        (np.tile([-1, 1], n_contacts), (row_ind, col_ind))
    ).tocsr()
