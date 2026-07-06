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


def build_reduction_operator(conductor: object) -> None:
    """Build the sparse Dirichlet reduction operator, once per topology build.

    The operator ``P`` (shape ``full_size x reduced_size``) encodes both
    boundary-condition reductions applied by :func:`reduce_system`:

    * **equipotential surfaces**: each redundant node of a group maps onto the
      column of the group's representative node, so ``P^T K P`` sums the
      merged rows and columns;
    * **fixed-potential nodes**: no column at all, so their rows and columns
      are dropped (their prescribed values are re-inserted into the solution
      by :func:`~electromagnetics.electric_solver.assemble_solution`).

    Since the node topology does not change with temperature or current, the
    operator is built once and :func:`reduce_system` only performs the cheap
    sparse products at every electric time step.

    Stores ``conductor.electric_reduction_operator`` (CSR) and
    ``conductor.electric_retained_index`` (the retained degree-of-freedom
    indices in full-size numbering).

    Args:
        conductor: Conductor object with fixed-potential and equipotential
            data already assigned.
    """
    # Deduplicate the fixed-potential DOFs (was previously redone at every
    # reduction call).
    conductor.fixed_potential_index, unique_idx = np.unique(
        conductor.fixed_potential_index, return_index=True
    )
    conductor.fixed_potential_value = conductor.fixed_potential_value[unique_idx]

    full_size = (
        conductor.total_elements_current_carriers
        + conductor.total_nodes_current_carriers
    )

    if conductor.operations.do_equipotential_surfaces_exist:
        redundant_index = conductor.equipotential_node_index[:, 1:].ravel()
    else:
        redundant_index = np.zeros(0, dtype=int)

    retained_index = np.setdiff1d(
        np.arange(full_size),
        np.unique(
            np.concatenate((conductor.fixed_potential_index, redundant_index))
        ),
    )

    column_of_dof = np.full(full_size, -1, dtype=int)
    column_of_dof[retained_index] = np.arange(retained_index.size)

    rows = [retained_index]
    columns = [np.arange(retained_index.size)]
    if conductor.operations.do_equipotential_surfaces_exist:
        for group in conductor.equipotential_node_index:
            rows.append(group[1:])
            columns.append(
                np.full(group[1:].size, column_of_dof[group[0]], dtype=int)
            )

    rows = np.concatenate(rows)
    columns = np.concatenate(columns)

    conductor.electric_reduction_operator = csr_matrix(
        (np.ones(rows.size), (rows, columns)),
        shape=(full_size, retained_index.size),
    )
    conductor.electric_retained_index = retained_index


def reduce_system(conductor: object) -> np.ndarray:
    """Apply Dirichlet BCs by reducing the stiffness matrix and RHS vector.

    Applies the cached reduction operator built by
    :func:`build_reduction_operator`::

        K_reduced = P^T K P
        b_reduced = P^T b

    which merges equipotential surface rows/columns (sum) and drops the
    rows/columns of all eliminated degrees of freedom, entirely in sparse
    arithmetic (the previous implementation densified the full system at
    every electric time step).

    The reduced ``conductor.electric_stiffness_matrix`` and
    ``conductor.electric_known_term_vector`` are stored back on the conductor.

    Args:
        conductor: Conductor object with the stiffness matrix and known-term
            vector filled in and the reduction operator built.

    Returns:
        Integer array of retained degree-of-freedom indices in the original
        (full-size) numbering.  Passed to
        :func:`~electromagnetics.electric_solver.assemble_solution` to place
        the reduced solution back into the full vector.
    """
    operator = conductor.electric_reduction_operator

    conductor.electric_stiffness_matrix = (
        operator.T @ conductor.electric_stiffness_matrix @ operator
    ).tocsr()
    conductor.electric_known_term_vector = (
        operator.T @ conductor.electric_known_term_vector
    )

    return conductor.electric_retained_index
