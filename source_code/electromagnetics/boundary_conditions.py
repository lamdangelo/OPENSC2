"""
This module applies Dirichlet boundary conditions to the electric linear
system before solving:

1. **Equipotential surfaces**: groups of nodes that are forced to share the
   same potential.  The corresponding rows and columns are merged (summed)
   in the stiffness matrix and known-term vector, then the redundant rows
   and columns are removed.

2. **Fixed-potential nodes**: individual nodes whose potential is prescribed.
   The column contribution is subtracted from the right-hand side and the
   corresponding row/column are eliminated.

:func:`assign_equipotential_surfaces` and :func:`assign_fixed_potential`
prepare the conductor's index/value arrays from the input data. They are
called once per thermal time step inside ``electric_preprocessing``.

:func:`reduce_system` performs the actual matrix reduction and is called
at every electric time step just before the linear solve.

Ported from the private methods of the Conductor class and the utility
function ``fixed_value`` in ``electric_auxiliary_functions``:
    __assign_equivalue_surfaces
    __assign_fix_potential
    fixed_value
"""

import numpy as np
from scipy.sparse import csr_matrix


def assign_equipotential_surfaces(conductor: object) -> None:
    """Fill ``conductor.equipotential_node_index`` from the operation settings.

    For each prescribed equipotential surface coordinate, finds the last
    ``n_strands`` nodes whose axial position does not exceed the coordinate
    (i.e. the cross-section just before or at that coordinate) and records
    their global indices (shifted by ``total_elements_current_carriers`` to
    address the potential partition of the solution vector).

    Does nothing if ``conductor.operations.do_equipotential_surfaces_exist``
    is ``False``.

    Args:
        conductor: Conductor object with nodal coordinates, mesh, inventory,
            and operations set up.
    """
    if not conductor.operations.do_equipotential_surfaces_exist:
        return

    n_el = conductor.total_elements_current_carriers
    n_strands = conductor.inventory.strands.number
    z_values = conductor.nodal_coordinates.loc["StrandComponent", "z"].to_numpy()

    for ii, coord in enumerate(
        conductor.operations.equipotential_surface_coordinates
    ):
        conductor.equipotential_node_index[ii, :] = (
            np.nonzero(z_values <= round(coord, conductor.mesh.position_precision))[0][
                -n_strands:
            ]
            + n_el
        )


def assign_fixed_potential(conductor: object) -> None:
    """Fill ``conductor.fixed_potential_index`` and ``conductor.fixed_potential_value``.

    Iterates over all strand objects that have a fixed potential prescribed
    (``ops.fix_potential_flag`` is truthy) and records the global node
    index (shifted by ``total_elements_current_carriers``) and the
    prescribed value for each such node.

    Args:
        conductor: Conductor object with nodal coordinates, inventory, and
            operation flags already set up.

    Raises:
        ValueError: if a prescribed z-coordinate does not match any node
            within tolerance ``1e-10``.
    """
    tol = 1e-10
    z_values = conductor.nodal_coordinates.xs("StrandComponent").z.to_numpy()
    offset = conductor.total_elements_current_carriers
    jj = 0

    for strand in conductor.inventory.strands.collection:
        ops = strand.operations
        if not ops.fix_potential_flag:
            continue

        n = ops.fix_potential_number
        conductor.fixed_potential_value[jj : jj + n] = ops.fix_potential_value

        for ii, coord in enumerate(ops.fix_potential_coordinate, start=jj):
            diff = np.abs(z_values - coord)
            match = np.flatnonzero(diff <= tol)
            if match.size == 0:
                raise ValueError(
                    f"No node found at z = {coord} for strand "
                    f"'{strand.identifier}' (tolerance {tol})."
                )
            conductor.fixed_potential_index[ii] = match[0] + offset

        jj += n


def reduce_system(conductor: object) -> np.ndarray:
    """Apply Dirichlet BCs by reducing the stiffness matrix and RHS vector.

    Performs the following steps in order:

    1. Subtract the fixed-potential column contributions from the RHS.
    2. Merge equipotential surface rows and columns (sum them).
    3. Delete the rows and columns of all eliminated degrees of freedom
       (both fixed and equipotential-redundant nodes).

    The reduced ``conductor.electric_stiffness_matrix`` and
    ``conductor.electric_known_term_vector`` are stored back on the conductor.

    Args:
        conductor: Conductor object with the stiffness matrix, known-term
            vector, fixed-potential data, and equipotential data filled in.

    Returns:
        ``idx``: integer array of retained degree-of-freedom indices in the
        original (full-size) numbering.  Passed to
        :func:`~electromagnetics.electric_solver.assemble_solution` to place
        the reduced solution back into the full vector.
    """
    # --- Step 1: subtract fixed-potential column contributions ---------------
    conductor.fixed_potential_index, unique_idx = np.unique(
        conductor.fixed_potential_index, return_index=True
    )
    conductor.fixed_potential_value = conductor.fixed_potential_value[unique_idx]

    if np.isscalar(conductor.fixed_potential_index):
        conductor.electric_known_term_vector -= (
            conductor.electric_stiffness_matrix[
                :, conductor.fixed_potential_index
            ]
            @ conductor.fixed_potential_value
        )

    # --- Step 2: merge equipotential surface rows/columns --------------------
    n_equipotential_nodes = conductor.equipotential_node_index.size
    n_surfaces = (
        conductor.operations.number_of_equipotential_surfaces
        if conductor.operations.do_equipotential_surfaces_exist
        else 0
    )
    removed_index = np.zeros(n_equipotential_nodes - n_surfaces, dtype=int)

    if conductor.operations.do_equipotential_surfaces_exist:
        for ii, row in enumerate(conductor.equipotential_node_index):
            # Sum columns (merge the equipotential nodes into the first one)
            conductor.electric_stiffness_matrix[:, row[0]] = np.sum(
                conductor.electric_stiffness_matrix[:, row], axis=1
            )
            removed_index[
                ii * row[1:].shape[0] : (ii + 1) * row[1:].shape[0]
            ] = row[1:]

            # Sum rows
            conductor.electric_stiffness_matrix[row[0], :] = np.sum(
                conductor.electric_stiffness_matrix[row, :], axis=0
            )
            conductor.electric_known_term_vector[row[0]] = np.sum(
                conductor.electric_known_term_vector[row]
            )

    # --- Step 3: delete eliminated rows/columns ------------------------------
    full_size = conductor.electric_known_term_vector.shape[0]
    idx = np.setdiff1d(
        np.r_[0:full_size],
        np.unique(
            np.concatenate((conductor.fixed_potential_index, removed_index))
        ),
        assume_unique=True,
    )

    conductor.electric_stiffness_matrix = csr_matrix(
        conductor.electric_stiffness_matrix.toarray()[np.ix_(idx, idx)]
    )
    conductor.electric_known_term_vector = conductor.electric_known_term_vector[idx]

    return idx
