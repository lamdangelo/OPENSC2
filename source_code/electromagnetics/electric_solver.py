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

import warnings

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu, spsolve

from electromagnetics.boundary_conditions import reduce_system
from electromagnetics.electromagnetic_flags import ElectricSolver
from electromagnetics.resistance import build_differential_resistance_matrix
from electromagnetics.system_assembly import build_known_term_vector, build_right_hand_side

# Default number of electric sub-steps per thermal time step.
ELECTRIC_TIME_STEP_NUMBER: int = 10


def _solve_steady_reduced(conductor, matrix, right_hand_side):
    """Solve the reduced steady-state system.

    For STEADY_STATE-flagged conductors (typically many-strand
    discretizations) the system is a long, thin ladder whose default
    COLAMD ordering suffers catastrophic fill-in (memory exhaustion at
    ~7e5 unknowns). A cached reverse Cuthill-McKee permutation turns it
    into a narrow band (bandwidth ~2 x strands) that factorises in
    seconds. TRANSIENT-flagged conductors keep the plain spsolve so
    their (single) step-0 steady solve stays bit-identical.
    """
    if conductor.operations.electric_solver != ElectricSolver.STEADY_STATE:
        return spsolve(matrix, right_hand_side)

    matrix = matrix.tocsr()
    permutation = getattr(conductor, "_steady_state_permutation", None)
    if permutation is None or permutation.size != matrix.shape[0]:
        structure = (abs(matrix) + abs(matrix).T).tocsr()
        permutation = reverse_cuthill_mckee(structure, symmetric_mode=True)
        conductor._steady_state_permutation = permutation

    permuted = matrix[permutation][:, permutation].tocsc()
    solution_permuted = splu(permuted, permc_spec="NATURAL").solve(
        np.asarray(right_hand_side)[permutation]
    )
    solution = np.empty_like(solution_permuted)
    solution[permutation] = solution_permuted
    return solution


# ---------------------------------------------------------------------------
# HIGH-LEVEL SOLVERS
# ---------------------------------------------------------------------------

# Convergence controls of the steady-state current-consistency Newton
# solver (ELECTRIC_CURRENT_CONSISTENCY). The residual is measured
# blockwise (element rows are volts, node rows amperes) against the
# natural scales of each block; the line search backtracks on the plain
# 2-norm (monotonicity only needs *some* norm).
STEADY_NEWTON_RELATIVE_TOLERANCE: float = 1.0e-8
STEADY_NEWTON_MAX_BACKTRACKS: int = 8
STEADY_NEWTON_ARMIJO: float = 1.0e-4


def _steady_solve_once(conductor: object) -> None:
    """One steady-state build-and-solve pass (matrices from the current
    ``current_for_resistance`` bindings)."""
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
    electric_solution_reduced = _solve_steady_reduced(
        conductor,
        conductor.electric_stiffness_matrix,
        conductor.electric_right_hand_side,
    )

    assemble_solution(conductor, idx, electric_solution_reduced)

    # Reset the known-term vector to its full (unreduced) size for the next call
    conductor.electric_known_term_vector = np.zeros(
        conductor.total_elements_current_carriers
        + conductor.total_nodes_current_carriers
    )


def _gauss_solved_current(strand: object) -> np.ndarray:
    """Solved strand current as a non-negative real array.

    ``current_along`` can be negative (sign convention: positive = +z)
    and complex (assemble_solution); the power-law resistivity needs a
    non-negative real base (a non-integer exponent on a negative base
    yields NaN).
    """
    return np.abs(np.real(strand.gauss_fields.current_along))


def _node_from_gauss(gauss: np.ndarray) -> np.ndarray:
    """Element-centred values -> nodal values (adjacent average, ends copied)."""
    nodal = np.empty(gauss.size + 1)
    nodal[1:-1] = 0.5 * (gauss[:-1] + gauss[1:])
    nodal[0] = gauss[0]
    nodal[-1] = gauss[-1]
    return nodal


def _bind_current_for_resistance(strand: object, gauss_current: np.ndarray) -> None:
    """Rebind the resistance-evaluation current of ``strand``.

    The ``_sc`` variants alias the same arrays, mirroring the FROM_FILE
    aliasing that makes the SC-regime in-place mutation in
    ``get_electric_resistance`` a no-op.
    """
    strand.gauss_fields.current_for_resistance = gauss_current
    strand.node_fields.current_for_resistance = _node_from_gauss(gauss_current)
    if hasattr(strand.gauss_fields, "op_current_sc"):
        strand.gauss_fields.current_for_resistance_sc = (
            strand.gauss_fields.current_for_resistance
        )
        strand.node_fields.current_for_resistance_sc = (
            strand.node_fields.current_for_resistance
        )


def _steady_voltage_scale_floor(conductor: object) -> float:
    """Natural voltage scale of the power law: max over superconducting
    strands of E0 times the largest element length. Prevents a zero
    denominator in the element-block convergence test when the whole
    conductor is superconducting (residual voltages ~ 0)."""
    floor = 0.0
    longest_element = float(np.max(conductor.mesh.element_lengths))
    for strand in conductor.inventory.strands.collection:
        electric_field_criterion = getattr(
            strand.inputs, "flux_flow_electric_field", None
        )
        if electric_field_criterion:
            floor = max(
                floor, float(electric_field_criterion) * longest_element
            )
    return max(floor, 1.0e-12)


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

    With ``ELECTRIC_CURRENT_CONSISTENCY`` set on the conductor, a damped
    Newton method solves the nonlinear network equations

        F(x) = K(|I|) x - b = 0,   x = [element currents; node potentials]

    so the strand resistances are exactly consistent with the currents
    that flow (the power-law V-I curve is odd, so R evaluated at |I|
    reproduces V(I) exactly). The Jacobian differs from K only in the
    diagonal resistance block, replaced by the differential resistance
    d(V_e)/d(I_e) (:func:`~electromagnetics.resistance.build_differential_resistance_matrix`),
    hence it shares K's sparsity and the cached RCM permutation. A
    backtracking line search on the reduced-residual norm globalizes the
    iteration; ``MAXIMUM_ITERATION_NUMBER`` caps it. On the cap a warning
    is emitted and the best iterate is kept (a hard error would abort
    multi-hour transients). Tcs/margin diagnostics (``eval_tcs``)
    deliberately stay on the imposed current: they run in
    ``operating_conditions_em`` before the loop. Fixed potentials are
    inserted into the state before every residual evaluation, which is
    the correct general form (the legacy reduction is only consistent
    for zero fixed values; all shipped decks use zero).

    The solution is stored in ``conductor.electric_solution`` and a copy is
    kept in ``conductor.electric_solution_steady`` for use as the initial
    condition of the transient loop (and as the Newton warm start of the
    next thermal step).

    Args:
        conductor: Conductor object whose operating conditions have already
            been set for the current thermal time step.
    """
    conductor.operating_conditions_em()

    if not conductor.operations.electric_current_consistency:
        # Legacy path: single build-and-solve from the imposed current.
        _steady_solve_once(conductor)
        conductor.electric_solution_steady = conductor.electric_solution.copy()
        return

    _solve_steady_newton(conductor)
    conductor.electric_solution_steady = conductor.electric_solution.copy()


def _solve_steady_newton(conductor: object) -> None:
    """Damped Newton iteration for the steady current-consistent solve.

    See :func:`solve_steady_state` for the formulation. Postconditions:
    ``conductor.electric_solution`` holds the adopted full-size state;
    the stiffness matrix, the strand ``electric_resistance`` fields and
    ``parallel_jacket_pairs`` are consistent with it (the last residual
    evaluation ran at the adopted state); the full-size
    ``electric_known_term_vector`` buffer invariant is preserved.
    """
    n_elements = conductor.total_elements_current_carriers
    full_size = n_elements + conductor.total_nodes_current_carriers
    strands = conductor.inventory.strands.collection
    n_strands = conductor.inventory.strands.number
    cap = max(1, int(conductor.operations.maximum_iteration_number))

    if np.iscomplexobj(np.asarray(conductor.node_fields.op_current)):
        raise ValueError(
            "ELECTRIC_CURRENT_CONSISTENCY requires a real steady-state "
            "system (complex current injections found)."
        )

    # Warm start: previous thermal step's solution, else one legacy solve
    # from the imposed-current resistances (also builds the topology and
    # the reduction operator on the first call).
    warm = getattr(conductor, "electric_solution_steady", None)
    if warm is None or np.asarray(warm).shape[0] != full_size:
        _steady_solve_once(conductor)
        warm = conductor.electric_solution
    x = np.real(np.asarray(warm)).astype(float).copy()

    def bind(x_full):
        currents = x_full[:n_elements]
        for ii, strand in enumerate(strands):
            _bind_current_for_resistance(strand, np.abs(currents[ii::n_strands]))

    def residual(x_full):
        """Bind |I|, rebuild R and K, return (reduced residual, P).

        P is re-fetched after preprocessing because ADAPTED/FROM_FILE
        meshes rebuild the topology (and the reduction operator) inside
        every ``electric_preprocessing`` call.
        """
        bind(x_full)
        conductor.electric_preprocessing()
        operator = conductor.electric_reduction_operator
        known = np.zeros(full_size)
        known[n_elements:] = conductor.node_fields.op_current
        return (
            operator.T @ (conductor.electric_stiffness_matrix @ x_full - known),
            operator,
        )

    def scatter(x_reduced, operator):
        """Reduced state -> full state (replicates equipotential group
        members and inserts the fixed potential values, exactly like
        ``assemble_solution``)."""
        x_full = operator @ x_reduced
        if conductor.fixed_potential_index.size > 0:
            x_full[conductor.fixed_potential_index] = (
                conductor.fixed_potential_value
            )
        return x_full

    def converged(residual_reduced, x_full):
        """Blockwise test: element rows are volts, node rows amperes."""
        retained = conductor.electric_retained_index
        element_rows = retained < n_elements
        residual_elements = residual_reduced[element_rows]
        residual_nodes = residual_reduced[~element_rows]
        current_scale = max(
            float(np.abs(conductor.node_fields.op_current).max()), 1.0
        )
        along_voltage = conductor.incidence_matrix @ x_full[n_elements:]
        voltage_scale = max(
            float(np.abs(along_voltage).max()) if along_voltage.size else 0.0,
            _steady_voltage_scale_floor(conductor),
        )
        nodes_ok = (
            float(np.abs(residual_nodes).max()) if residual_nodes.size else 0.0
        ) <= STEADY_NEWTON_RELATIVE_TOLERANCE * current_scale
        elements_ok = (
            float(np.abs(residual_elements).max())
            if residual_elements.size
            else 0.0
        ) <= STEADY_NEWTON_RELATIVE_TOLERANCE * voltage_scale
        return nodes_ok and elements_ok

    # Canonicalize the warm start through the reduction operator (the
    # first residual() call also builds the topology if needed).
    residual_reduced, operator = residual(x)
    x = scatter(x[conductor.electric_retained_index], operator)
    residual_reduced, operator = residual(x)
    best_norm = float(np.linalg.norm(residual_reduced))
    best_state = x.copy()

    iteration = 0
    for iteration in range(1, cap + 1):
        if converged(residual_reduced, x):
            break

        # Jacobian: same sparsity as K, only the resistance block differs.
        build_differential_resistance_matrix(conductor)
        jacobian = bmat(
            [
                [
                    conductor.electric_differential_resistance_matrix,
                    conductor.incidence_matrix,
                ],
                [
                    -conductor.incidence_matrix_transposed,
                    conductor.electric_conductance_matrix,
                ],
            ],
            format="csr",
            dtype=float,
        )
        try:
            delta = _solve_steady_reduced(
                conductor,
                (operator.T @ jacobian @ operator).tocsr(),
                -residual_reduced,
            )
        except RuntimeError as error:
            warnings.warn(
                "Steady electric current-consistency Newton step failed "
                f"(singular Jacobian factorization: {error}); keeping the "
                "best iterate."
            )
            break

        # Backtracking line search on the reduced-residual 2-norm; the
        # trial residual is always the true residual (full resistance
        # rebuild per trial - no factorization, so trials are cheap).
        norm_previous = float(np.linalg.norm(residual_reduced))
        x_reduced = x[conductor.electric_retained_index]
        step = 1.0
        for _ in range(STEADY_NEWTON_MAX_BACKTRACKS + 1):
            x_trial = scatter(x_reduced + step * delta, operator)
            residual_trial, operator_trial = residual(x_trial)
            if float(np.linalg.norm(residual_trial)) <= (
                1.0 - STEADY_NEWTON_ARMIJO * step
            ) * norm_previous:
                break
            step *= 0.5
        # Adopt the last trial even on a stalled search: the cap plus
        # warning is the failure semantics, and the best iterate is
        # tracked separately.
        x, residual_reduced, operator = x_trial, residual_trial, operator_trial
        norm_current = float(np.linalg.norm(residual_reduced))
        if norm_current < best_norm:
            best_norm = norm_current
            best_state = x.copy()
    else:
        x = best_state
        residual_reduced, operator = residual(x)
        warnings.warn(
            "Steady electric current-consistency Newton iteration did not "
            f"converge in {cap} iterations (best residual norm "
            f"{best_norm:.3e}); continuing with the best iterate."
        )

    # The last residual() evaluation ran at the adopted x, so K, the
    # strand electric_resistance fields and parallel_jacket_pairs are
    # consistent with the solution: Joule power downstream is exact.
    conductor.electric_solution = x
    conductor._steady_newton_iterations = iteration
    # Restore the full-size known-term buffer invariant (transient path
    # sizes itself from this attribute).
    conductor.electric_known_term_vector = np.zeros(full_size)

    conductor.electric_solution_steady = conductor.electric_solution.copy()


def solve_transient(conductor: object) -> None:
    """Run the transient electric sub-stepping loop within one thermal step.

    Future work: the transient path still evaluates the strand
    resistances from the imposed operating current even when
    ``ELECTRIC_CURRENT_CONSISTENCY`` is set; the consistency iteration
    currently applies only to the steady-state solve.

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

    # Number of electric sub-steps needed to cover the thermal time step
    # (the loop was previously hardcoded to ELECTRIC_TIME_STEP_NUMBER
    # iterations regardless of the user-defined electric time step).
    number_of_substeps = max(
        1, round(conductor.electric_time_end / conductor.electric_time_step)
    )

    for nn in range(1, number_of_substeps + 1):
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

    # NOTE: an AC-RMS convention (amplitude / sqrt(2) on voltage and
    # current, i.e. halved power) used to be applied here for
    # STEADY_STATE-flagged conductors. The steady solve models DC
    # quasi-static current sharing in this code base, where P = V * I
    # holds without the factor, so the halving was removed (no shipped
    # deck used the STEADY_STATE flag).

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
