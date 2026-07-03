"""
This module contains the high-level electric problem solvers:

* :func:`solve_steady_state`: solves the electric problem once for the
  initial, quasi-static operating point.
* :func:`solve_transient`: runs the inner electric time-stepping loop
  within a single thermal time step, using the theta-method (Backward
  Euler, Crank-Nicolson, or Adams-Moulton 4th order as set in the inputs).

Both functions drive the preprocessing, boundary-condition application,
linear solve, and solution post-processing steps by calling the appropriate
functions in the other electromagnetics sub-modules.

Post-processing utilities that act on the solved electric solution:
* :func:`reorganize_solution`: extracts per-strand currents, nodal
  potentials, and voltage differences from the flat solution vector.
* :func:`evaluate_joule_power_conductance`: computes the Joule power
  dissipated in the transverse electric conductances between strands.
* :func:`compute_voltage_sum`: integrates voltage drops along the
  conductor between pairs of diagnostic spatial coordinates.

:data:`ELECTRIC_TIME_STEP_NUMBER` sets the default number of electric
sub-steps taken within each thermal time step.

Ported from:
    utility_functions/electric_auxiliary_functions.py:
        electric_steady_state_solution
        electric_transient_solution
        solution_completion
        fixed_value  (now in boundary_conditions.py)
    Conductor private methods:
        __electric_solution_reorganization
        __get_total_joule_power_electric_conductance
        __compute_voltage_sum
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from electromagnetics.boundary_conditions import reduce_system
from electromagnetics.system_assembly import build_known_term_vector, build_right_hand_side

# Default number of electric sub-steps per thermal time step.
ELECTRIC_TIME_STEP_NUMBER: int = 10


# ---------------------------------------------------------------------------
# HIGH-LEVEL SOLVERS
# ---------------------------------------------------------------------------

def solve_steady_state(conductor: object) -> None:
    """Solve the electric problem in steady-state (quasi-static) conditions.

    Sequence:
    1. Update electromagnetic operating conditions (B-field, current,
       critical properties) in Gauss points.
    2. Rebuild all electric matrices (resistance, conductance, stiffness).
    3. Assemble the known-term vector.
    4. Reduce the system (apply Dirichlet BCs).
    5. Solve the reduced sparse linear system.
    6. Reconstruct the full solution vector.

    The solution is stored in ``conductor.electric_solution`` and a copy is
    kept in ``conductor.electric_solution_steady`` for use as the initial
    condition of the transient loop.

    Args:
        conductor: Conductor object whose operating conditions have already
            been set for the current thermal time step.
    """
    conductor.operating_conditions_em()
    conductor.electric_preprocessing()

    n_size = conductor.electric_known_term_vector.shape[0]
    if n_size == 0:
        conductor.electric_known_term_vector = np.zeros(
            conductor.electric_stiffness_matrix.shape[0]
        )

    conductor.electric_solution = np.zeros(
        conductor.electric_known_term_vector.shape[0]
    )

    build_known_term_vector(conductor)
    idx = reduce_system(conductor)

    # Use the known-term vector as the RHS (steady-state has no mass-matrix term)
    conductor.electric_right_hand_side = conductor.electric_known_term_vector
    electric_solution_reduced = spsolve(
        conductor.electric_stiffness_matrix,
        conductor.electric_right_hand_side,
        permc_spec="NATURAL",
    )

    assemble_solution(conductor, idx, electric_solution_reduced)

    # Reset the known-term vector to its full (unreduced) size for the next call
    conductor.electric_known_term_vector = np.zeros(
        conductor.total_elements_current_carriers
        + conductor.total_nodes_current_carriers
    )

    conductor.electric_solution_steady = conductor.electric_solution.copy()


def solve_transient(conductor: object) -> None:
    """Run the transient electric sub-stepping loop within one thermal step.

    Iterates ``ELECTRIC_TIME_STEP_NUMBER`` sub-steps of size
    ``conductor.electric_time_step``.  At each sub-step:

    1. Update electromagnetic operating conditions.
    2. Rebuild the resistance matrix (only resistance changes with T).
    3. Assemble the transient stiffness and mass matrices.
    4. Assemble and reduce the RHS (theta-method blending + old solution
       contribution).
    5. Solve the reduced system.
    6. Reconstruct the full solution.

    At sub-step 0, the steady-state solution is used as the initial condition.

    Args:
        conductor: Conductor object with ``electric_solution_steady``,
            ``electric_mass_matrix``, ``electric_theta``, and
            ``electric_time_step`` set up.
    """
    n_total = (
        conductor.total_elements_current_carriers
        + conductor.total_nodes_current_carriers
    )
    if conductor.electric_known_term_vector.shape[0] == 0:
        conductor.electric_known_term_vector = np.zeros_like(
            conductor.electric_solution_steady
        )

    if conductor.cond_el_num_step == 0:
        conductor.electric_solution = conductor.electric_solution_steady.copy()

    build_known_term_vector(conductor)
    conductor.electric_known_term_vector_old = (
        conductor.electric_known_term_vector.copy()
    )

    for nn in range(1, ELECTRIC_TIME_STEP_NUMBER + 1):
        conductor.electric_time += conductor.electric_time_step
        conductor.cond_el_num_step = nn

        conductor.operating_conditions_em()
        conductor.eval_total_operating_current()
        conductor.electric_preprocessing()

        # Form the transient stiffness matrix: M/dt + theta*K
        K_static = conductor.electric_stiffness_matrix.copy()
        conductor.electric_stiffness_matrix = (
            conductor.electric_mass_matrix / conductor.electric_time_step
            + conductor.electric_theta * K_static
        )

        # The "foo" matrix contributes M/dt * x_old - (1-theta)*K*x_old to the RHS
        foo = (
            conductor.electric_mass_matrix / conductor.electric_time_step
            - (1.0 - conductor.electric_theta) * K_static
        )

        # Initialise known-term to zero before reduction (required by reduce_system)
        conductor.electric_known_term_vector = np.zeros_like(
            conductor.electric_solution_steady
        )
        idx = reduce_system(conductor)
        electric_known_term_reduced = conductor.electric_known_term_vector.copy()

        # Restore full-size known-term and rebuild for this sub-step
        conductor.electric_known_term_vector = np.zeros(
            np.zeros_like(conductor.electric_known_term_vector_old).shape
        )
        build_known_term_vector(conductor)

        build_right_hand_side(conductor, foo, electric_known_term_reduced, idx)

        electric_solution_reduced = spsolve(
            conductor.electric_stiffness_matrix,
            conductor.electric_right_hand_side,
            permc_spec="NATURAL",
        )

        conductor.electric_known_term_vector_old = (
            conductor.electric_known_term_vector.copy()
        )
        assemble_solution(conductor, idx, electric_solution_reduced)


# ---------------------------------------------------------------------------
# SOLUTION ASSEMBLY
# ---------------------------------------------------------------------------

def assemble_solution(
    conductor: object, idx: np.ndarray, electric_solution_reduced: np.ndarray
) -> None:
    """Place the reduced solution back into the full-size solution vector.

    Handles complex-valued solutions, inserts fixed-potential values, and
    replicates equipotential surface values to redundant nodes.

    Args:
        conductor: Conductor object with ``electric_solution``,
            ``fixed_potential_index``, ``fixed_potential_value``, and
            ``equipotential_node_index`` attributes.
        idx: Array of retained DOF indices (as returned by
            :func:`~electromagnetics.boundary_conditions.reduce_system`).
        electric_solution_reduced: Solution vector for the reduced system.
    """
    if np.iscomplex(electric_solution_reduced).any():
        conductor.electric_solution = conductor.electric_solution.astype(complex)

    conductor.electric_solution[idx] = electric_solution_reduced

    if conductor.fixed_potential_index.size > 0:
        conductor.electric_solution[conductor.fixed_potential_index] = (
            conductor.fixed_potential_value
        )

    if conductor.operations.do_equipotential_surfaces_exist:
        for _, row in enumerate(conductor.equipotential_node_index):
            conductor.electric_solution[row[1:]] = conductor.electric_solution[row[0]]


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

def reorganize_solution(conductor: object) -> None:
    """Distribute the flat electric solution vector to per-strand arrays.

    Splits ``conductor.electric_solution`` into:
    * **Edge currents** (first ``n_el`` entries): stored in each strand's
      ``gauss_fields.current_along``.
    * **Nodal potentials** (last ``n_nd`` entries): stored in
      ``conductor.nodal_potential``.

    Also computes:
    * ``delta_voltage_along``: longitudinal voltage drop per element
      (``-A * phi``), stored per strand in ``gauss_fields``.
    * ``delta_voltag_along_R``: resistive voltage drop (``I * R``) per
      element, stored per strand in ``gauss_fields``.

    Args:
        conductor: Conductor object with the electric solution and inventory
            already set up.
    """
    n_el = conductor.total_elements_current_carriers
    n_strands = conductor.inventory.strands.number

    current_along = conductor.electric_solution[:n_el]
    conductor.nodal_potential = conductor.electric_solution[n_el:]

    delta_voltage_along = -conductor.incidence_matrix @ conductor.nodal_potential

    for ii, strand in enumerate(conductor.inventory.strands.collection):
        strand.gauss_fields.current_along = current_along[ii::n_strands]
        strand.gauss_fields.delta_voltage_along = delta_voltage_along[ii::n_strands]
        strand.gauss_fields.delta_voltag_along_R = (
            strand.gauss_fields.current_along
            * strand.gauss_fields.electric_resistance
        )


def evaluate_joule_power_conductance(conductor: object) -> None:
    """Compute the Joule power dissipated in transverse electric conductances.

    For conductors with more than one strand, the transverse current flowing
    through each contact conductance is computed from the nodal potentials
    and the contact incidence matrix.  The resulting power is halved (since
    each contact appears in two nodes) and distributed to each strand's
    ``node_fields.total_power_el_cond``.

    For a single-strand conductor the power is zero and only the
    initialisation is performed.

    In the steady-state (sinusoidal) regime the instantaneous voltages and
    currents are converted to effective (RMS) values by dividing by ``sqrt(2)``.

    Args:
        conductor: Conductor object with nodal potential, contact incidence
            matrix, and conductance diagonal matrix set up.
    """
    from electromagnetics.electromagnetic_flags import ElectricSolver

    for strand in conductor.inventory.strands.collection:
        strand.node_fields.total_power_el_cond = 0.0

    if conductor.inventory.strands.number <= 1:
        return

    delta_voltage_across = np.real(
        conductor.contact_incidence_matrix @ conductor.nodal_potential
    )
    current_across = np.real(
        np.conj(
            conductor.electric_conductance_diag_matrix
            @ conductor.contact_incidence_matrix
            @ conductor.nodal_potential
        )
    )

    if conductor.operations.electric_solver == ElectricSolver.STEADY_STATE:
        delta_voltage_across /= np.sqrt(2.0)
        current_across /= np.sqrt(2.0)

    joule_power_across = delta_voltage_across * current_across
    joule_power_per_node = (
        np.abs(conductor.contact_incidence_matrix.T) @ joule_power_across
    ) / 2

    n_strands = conductor.inventory.strands.number
    for ii, strand in enumerate(conductor.inventory.strands.collection):
        strand.node_fields.total_power_el_cond = joule_power_per_node[
            ii::n_strands
        ]

    conductor.delta_voltage_across = delta_voltage_across
    conductor.current_across = current_across


def compute_voltage_sum(conductor: object) -> None:
    """Integrate the longitudinal voltage drop between diagnostic coordinates.

    For each pair of consecutive diagnostic axial positions (stored in
    ``conductor.Time_save``), the cumulative voltage drop is summed over all
    elements between those positions and stored at the right-most diagnostic
    point in each strand's ``gauss_fields.delta_voltage_along_sum``.

    Args:
        conductor: Conductor object with ``Time_save``, mesh Gauss point
            positions, and strand ``gauss_fields`` populated.
    """
    ind_zcoord_gauss = np.array(
        [0]
        + [
            np.max(
                np.nonzero(
                    conductor.mesh.gauss_point_coordinates
                    <= round(
                        conductor.Time_save[ii], conductor.mesh.position_precision
                    )
                )
            )
            for ii in range(1, conductor.Time_save.size)
        ]
    )

    for strand in conductor.inventory.strands.collection:
        for ii, idx in enumerate(ind_zcoord_gauss, 1):
            if ii < ind_zcoord_gauss.shape[0]:
                idxp = ind_zcoord_gauss[ii]
                strand.gauss_fields.delta_voltage_along_sum[idxp] = (
                    strand.gauss_fields.delta_voltage_along[idx : idxp + 1].sum()
                )
