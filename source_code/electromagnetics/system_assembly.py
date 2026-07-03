"""
This module assembles the global sparse matrices and vectors that define the
electric problem:

* **Stiffness matrix** ``K``:  couples edge currents to nodal potentials via
  the resistance and incidence matrices, and couples nodal potentials to each
  other via the transverse conductance matrix.

* **Mass matrix** ``M``:  encodes the inductance (for the transient term
  ``L dI/dt``) in the upper-left block; the remaining blocks are zero in the
  current formulation.

* **Known-term vector** ``f``:  right-hand side vector encoding the applied
  transport current boundary conditions.

* **Right-hand side** (transient):  the full ``b`` vector for the transient
  time-stepping scheme, assembled from the known-term vector, the mass
  matrix, and the solution at the previous time step.

Ported from the private methods of the Conductor class:
    __build_electric_stiffness_matrix
    __build_electric_mass_matrix
    build_electric_known_term_vector
    build_right_hand_side
"""

import numpy as np
from scipy.sparse import lil_matrix

from electromagnetics.inductance import build_inductance_matrix


def build_stiffness_matrix(conductor: object) -> None:
    """Assemble the electric stiffness matrix ``K``.

    The matrix has the block structure::

        K = [ R    A  ]
            [ -A^T G  ]

    where:
    * ``R``   = diagonal resistance matrix (shape ``n_el x n_el``);
    * ``A``   = incidence matrix            (shape ``n_el x n_nd``);
    * ``A^T`` = transposed incidence matrix (shape ``n_nd x n_el``);
    * ``G``   = nodal conductance matrix    (shape ``n_nd x n_nd``).

    The assembled matrix is stored in ``conductor.electric_stiffness_matrix``
    as a CSR sparse matrix.

    Note: The matrix is re-created from scratch at every call because the
    resistance matrix changes at each thermal time step as material properties
    evolve with temperature.

    Args:
        conductor: Conductor object with resistance, incidence, and
            conductance matrices already built.
    """
    n_el = conductor.total_elements_current_carriers
    n_nd = conductor.total_nodes_current_carriers
    size = n_el + n_nd

    K = lil_matrix((size, size), dtype=float)
    K[:n_el, :n_el] = conductor.electric_resistance_matrix
    K[:n_el, n_el:] = conductor.incidence_matrix
    K[n_el:, :n_el] = -conductor.incidence_matrix_transposed
    K[n_el:, n_el:] = conductor.electric_conductance_matrix

    conductor.electric_stiffness_matrix = K.tocsr(copy=True)


def build_mass_matrix(conductor: object) -> None:
    """Assemble the electric mass matrix ``M`` (inductance block).

    Calls :func:`~electromagnetics.inductance.build_inductance_matrix` to
    compute the full inductance matrix (self + mutual), then places it in the
    upper-left block of the mass matrix::

        M = [ L  0 ]
            [ 0  0 ]

    The assembled matrix is stored in ``conductor.electric_mass_matrix`` as a
    CSR sparse matrix.

    Args:
        conductor: Conductor object with inductance mode and self-inductance
            mode set in ``conductor.operations``.

    Raises:
        ValueError: propagated from :func:`build_inductance_matrix` if the
            inductance mode is not recognised.
    """
    build_inductance_matrix(conductor)

    n_el = conductor.total_elements_current_carriers
    conductor.electric_mass_matrix[:n_el, :n_el] = conductor.inductance_matrix
    conductor.electric_mass_matrix = conductor.electric_mass_matrix.tocsr(copy=True)


def build_known_term_vector(conductor: object) -> None:
    """Fill the electric known-term (right-hand side) vector with current sources.

    Copies the operating current vector into the nodal partition of
    ``conductor.electric_known_term_vector``. The current vector
    ``conductor.node_fields.op_current`` must already be evaluated by
    :func:`~electromagnetics.operating_conditions.evaluate_total_operating_current`.

    Args:
        conductor: Conductor object with ``node_fields.op_current`` and
            ``electric_known_term_vector`` initialised.
    """
    n_el = conductor.total_elements_current_carriers
    conductor.electric_known_term_vector[n_el:] = conductor.node_fields.op_current


def build_right_hand_side(
    conductor: object,
    foo: np.ndarray,
    bar: np.ndarray,
    idx: np.ndarray,
) -> None:
    """Build the right-hand side vector for the transient electric time step.

    The RHS combines the theta-method blending of the current and previous
    known-term vectors with the contribution from the old solution::

        rhs = theta * f_new + (1 - theta) * f_old + M/dt * x_old

    where the ``foo`` matrix already encodes ``M/dt - (1-theta)*K`` (computed
    in the solver).

    Equipotential surface contributions are summed before the index reduction
    is applied.

    Result stored in ``conductor.electric_right_hand_side``.

    Args:
        conductor: Conductor object with ``electric_theta``,
            ``electric_known_term_vector``, ``electric_known_term_vector_old``,
            and ``electric_solution`` attributes set.
        foo: Matrix ``M/dt - (1 - theta) * K`` of shape ``(size, size)``.
        bar: Reduced known-term vector after boundary-condition application,
            shape ``(size_reduced,)``.
        idx: Index array mapping reduced system positions back to full-size
            positions.
    """
    conductor.electric_right_hand_side = (
        conductor.electric_theta * conductor.electric_known_term_vector
        + (1.0 - conductor.electric_theta) * conductor.electric_known_term_vector_old
        + foo @ conductor.electric_solution
    )

    if conductor.operations.do_equipotential_surfaces_exist:
        for _, row in enumerate(conductor.equipotential_node_index):
            conductor.electric_right_hand_side[row[0]] = np.sum(
                conductor.electric_right_hand_side[row]
            )

    conductor.electric_right_hand_side = conductor.electric_right_hand_side[idx] - bar
