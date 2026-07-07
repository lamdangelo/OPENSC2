"""Standalone solver of the lumped-parameter hydraulic network (stage 0 of
the hydraulic field-circuit coupling).

Formulation: hydraulic modified nodal analysis in the unknown vector

    x = [pressure at the internal nodes | mass flow rate in the branches]

with the reservoir pressures acting as fixed sources. Mass balance of
internal node i (the KCL analog, with the compliance of a real-fluid
volume):

    C_i dp_i/dt + sum_b N[i, b] * mdot_b = q_i,   C_i = V_i * (rho*kappa_T)_old

and momentum balance of branch b oriented from node u to node v (the KVL +
resistance/inertance analog):

    L_b dmdot_b/dt + R_b * mdot_b - (p_u - p_v) = dp_source_b

where N is the oriented incidence matrix (+1 leaving, -1 entering), L_b the
inertance (length/cross_section for pipes; the density cancels in the
mass-flow formulation) and R_b the Picard-linearized resistance. Every
coefficient is frozen at the previous state -- density, viscosity and
isothermal compressibility through the shared CoolProp interface (same
cached property table as the channels), the friction factor at the previous
Reynolds number times the previous flow magnitude -- exactly the
linearize-at-old-state pattern of the 1D solver (see
hydraulics.momentum_equation.build_smat_fluid_momentum). One linear solve
advances a time step; there is no sub-iteration.

Time discretization mirrors the conductor (theta family and variable-step
BDF2, selected with conductor_flags.MethodFlag). Rows without a storage
term (zero-volume junctions, zero-inertance branches) are algebraic
constraints of the resulting DAE and are always enforced fully implicitly,
so that e.g. Crank-Nicolson cannot sustain an oscillating constraint
residual.
"""

from dataclasses import dataclass

import numpy as np

import interfaces.coolprop_interface as cpi
from conductor.conductor_flags import MethodFlag
from hydraulics.friction_factor_factory import FrictionFactorFactory
from hydraulics.friction_factor_models import FrictionFactorModelType
from hydraulics.network.network_inputs import (
    HydraulicNetworkInput,
    NetworkBranchKind,
    NetworkNodeKind,
)

# Reynolds number floor of the pipe resistance evaluation. In the laminar
# regime the product f(Re) * mdot is constant (f ~ 1/Re), so any floor
# inside that regime returns the exact laminar (Hagen-Poiseuille)
# resistance; it also keeps the resistance nonzero at zero-flow startup.
LAMINAR_REYNOLDS_FLOOR = 10.0

# Flow seeded into branches that start the steady-state Picard iteration at
# exactly zero flow: a purely quadratic resistance (or pure droop pump)
# linearized at zero flow would make the first iteration matrix singular.
STEADY_STATE_FLOW_SEED = 1.0e-3  # kg/s

# Same theta values as Conductor.__initialize_attributes; BDF2 uses its own
# multi-level coefficients and Adams-Moulton is not supported here.
_THETA_BY_METHOD = {
    MethodFlag.BACKWARD_EULER: 1.0,
    MethodFlag.CRANK_NICOLSON: 0.5,
    MethodFlag.GALERKIN: 2.0 / 3.0,
}

_NODE_PROPERTY_ALIASES = {
    "total_density": "Dmass",
    "isothermal_compressibility": "isothermal_compressibility",
}
_BRANCH_PROPERTY_ALIASES = {
    "total_density": "Dmass",
    "total_dynamic_viscosity": "viscosity",
}


@dataclass
class _PipeFrictionGeometry:
    """Duck-typed stand-in for FluidComponentInputs carrying exactly the
    attributes FrictionFactorFactory reads."""
    friction_factor_model: FrictionFactorModelType
    hydraulic_diameter: float
    roughness: float
    cross_section: float
    is_rectangular: bool = False
    width: float = 0.0
    height: float = 0.0
    void_fraction: float = 0.0


class HydraulicNetwork:
    """Topology, state and one-step solver of the lumped hydraulic network.

    Public solve entry points:
        * :meth:`solve_steady_state`: Picard iteration on the steady system,
          used to initialize the network flow (the network analog of
          gen_flow).
        * :meth:`step`: advance one time step (freeze properties at the old
          state, assemble, one linear solve).
    The split methods :meth:`update_properties`, :meth:`assemble_transient`
    and :meth:`advance` are exposed for the stage-1 bordered (Schur
    complement) solve, which needs the network blocks without an internal
    solve.
    """

    def __init__(
        self,
        inputs: HydraulicNetworkInput,
        method: MethodFlag = MethodFlag.BACKWARD_EULER,
    ):
        self.inputs = inputs
        self.fluid_type = inputs.fluid_type
        self.method = method
        if (
            method is not MethodFlag.BACKWARD_DIFFERENCE_2
            and method not in _THETA_BY_METHOD
        ):
            raise ValueError(
                f"Unsupported time integration method {method!r} for the "
                "hydraulic network."
            )
        self.theta = _THETA_BY_METHOD.get(method, 1.0)

        # Topology.
        self.node_inputs = list(inputs.nodes)
        self.branch_inputs = list(inputs.branches)
        self.node_index = {
            node.identifier: k for k, node in enumerate(self.node_inputs)
        }
        self.branch_index = {
            branch.identifier: k for k, branch in enumerate(self.branch_inputs)
        }
        number_of_nodes = len(self.node_inputs)
        number_of_branches = len(self.branch_inputs)
        # Oriented incidence matrix: +1 where the branch leaves the node,
        # -1 where it enters it.
        self.incidence = np.zeros((number_of_nodes, number_of_branches))
        self._branch_from = np.zeros(number_of_branches, dtype=int)
        self._branch_to = np.zeros(number_of_branches, dtype=int)
        for b, branch in enumerate(self.branch_inputs):
            self._branch_from[b] = self.node_index[branch.from_node]
            self._branch_to[b] = self.node_index[branch.to_node]
            self.incidence[self._branch_from[b], b] = 1.0
            self.incidence[self._branch_to[b], b] = -1.0
        self.internal_node_indices = np.array(
            [
                k for k, node in enumerate(self.node_inputs)
                if node.kind is NetworkNodeKind.INTERNAL
            ],
            dtype=int,
        )
        self.reservoir_node_indices = np.array(
            [
                k for k, node in enumerate(self.node_inputs)
                if node.kind is NetworkNodeKind.RESERVOIR
            ],
            dtype=int,
        )
        self.number_of_unknowns = (
            self.internal_node_indices.size + number_of_branches
        )

        # State (reservoir entries of node_pressure stay at their fixed
        # values; only the internal entries are unknowns).
        self.node_pressure = np.array(
            [node.pressure for node in self.node_inputs]
        )
        self.node_temperature = np.array(
            [node.temperature for node in self.node_inputs]
        )
        self.node_volume = np.array([node.volume for node in self.node_inputs])
        self.branch_mass_flow = np.array(
            [branch.initial_mass_flow for branch in self.branch_inputs]
        )
        self.branch_inertance = np.array(
            [branch.resolved_inertance() for branch in self.branch_inputs]
        )

        # Channel-style friction models of the pipe branches.
        self._friction_models = {
            b: FrictionFactorFactory.create(
                _PipeFrictionGeometry(
                    friction_factor_model=branch.friction_factor_model,
                    hydraulic_diameter=branch.hydraulic_diameter,
                    roughness=branch.roughness,
                    cross_section=branch.cross_section,
                )
            )
            for b, branch in enumerate(self.branch_inputs)
            if branch.kind is NetworkBranchKind.PIPE
        }

        # External mass sources into internal nodes, by node identifier
        # (kg/s, positive into the node); used for genuinely prescribed
        # sources and for the steady initialization with frozen port flows.
        self.external_mass_sources = {}

        # Identifiers of internal nodes whose mass balance receives an
        # implicit conductor port flow (filled by the port resolution).
        # Their rows are enforced fully implicitly by assemble_transient:
        # the port term has no old-time-level counterpart in the network
        # known term, so a theta blend of the rest of the row would mix
        # time levels inconsistently at the interface.
        self.coupled_node_identifiers = set()

        # Frozen properties, filled by update_properties.
        self.node_capacitance = np.zeros(number_of_nodes)
        self.branch_density = np.zeros(number_of_branches)
        self.branch_dynamic_viscosity = np.zeros(number_of_branches)
        self.update_properties()

        # Time integration history: two solution levels (BDF2 and the local
        # truncation error estimation of a later stage read the second one).
        self._solution_history = np.tile(self._pack_state()[:, None], (1, 2))
        self._last_source = None
        self._previous_source = None
        self.num_step = 0
        self.previous_time_step = 0.0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(fluid: {self.fluid_type.value}, "
            f"{len(self.node_inputs)} nodes, "
            f"{len(self.branch_inputs)} branches)"
        )

    def node_pressure_of(self, identifier: str) -> float:
        return float(self.node_pressure[self.node_index[identifier]])

    def branch_mass_flow_of(self, identifier: str) -> float:
        return float(self.branch_mass_flow[self.branch_index[identifier]])

    def state_vector(self) -> np.ndarray:
        """Current state in unknown-vector layout
        [internal node pressures | branch mass flows]."""
        return self._pack_state()

    def internal_node_unknown_index(self, identifier: str) -> int:
        """Position of an internal node's pressure in the unknown vector
        (which is also its equation row); raises for reservoirs."""
        node_index = self.node_index[identifier]
        positions = np.nonzero(self.internal_node_indices == node_index)[0]
        if positions.size == 0:
            raise ValueError(
                f"Network node {identifier!r} is not an internal node."
            )
        return int(positions[0])

    def _pack_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.node_pressure[self.internal_node_indices],
                self.branch_mass_flow,
            )
        )

    def _unpack_state(self, solution: np.ndarray):
        number_of_internal = self.internal_node_indices.size
        self.node_pressure[self.internal_node_indices] = solution[
            :number_of_internal
        ]
        self.branch_mass_flow = solution[number_of_internal:].copy()

    def update_properties(self):
        """Freeze the fluid properties at the current (old) state: node
        compliance C = V * rho * kappa_T and branch density/viscosity.

        The branch state point is the mean pressure of its end nodes at the
        temperature of the upstream node (by the current flow sign; the
        from-node on a tie). Same CoolProp interface and cached property
        table as the channel coolant."""
        node_properties = cpi.compute_properties(
            self.fluid_type,
            _NODE_PROPERTY_ALIASES,
            self.node_temperature,
            self.node_pressure,
        )
        self.node_capacitance = (
            self.node_volume
            * node_properties["total_density"]
            * node_properties["isothermal_compressibility"]
        )
        upstream = np.where(
            self.branch_mass_flow >= 0.0, self._branch_from, self._branch_to
        )
        branch_properties = cpi.compute_properties(
            self.fluid_type,
            _BRANCH_PROPERTY_ALIASES,
            self.node_temperature[upstream],
            0.5
            * (
                self.node_pressure[self._branch_from]
                + self.node_pressure[self._branch_to]
            ),
        )
        self.branch_density = branch_properties["total_density"]
        self.branch_dynamic_viscosity = branch_properties[
            "total_dynamic_viscosity"
        ]

    def _branch_resistance_and_source(self) -> tuple:
        """Picard-linearized resistance R_b and pressure source dp_source_b
        of every branch, frozen at the previous flow and properties.

        Pipe: dp = 2 f l rho v|v| / D_h in the flow variable, so
        R = 2 f(Re_old) l |mdot_old| / (D_h rho A^2), the lumped twin of the
        2 f |v| / D_h diagonal of build_smat_fluid_momentum (the friction
        multiplier applies to the total factor, as in
        Channel.evaluate_friction_factors). Pump: the droop terms act as a
        (stabilizing) resistance and the zero-flow head as a source."""
        number_of_branches = len(self.branch_inputs)
        resistance = np.zeros(number_of_branches)
        source = np.zeros(number_of_branches)
        for b, branch in enumerate(self.branch_inputs):
            flow_magnitude = abs(self.branch_mass_flow[b])
            if branch.kind is NetworkBranchKind.PIPE:
                area = branch.cross_section
                diameter = branch.hydraulic_diameter
                viscosity = self.branch_dynamic_viscosity[b]
                flow_magnitude = max(
                    flow_magnitude,
                    LAMINAR_REYNOLDS_FLOOR * area * viscosity / diameter,
                )
                reynolds = flow_magnitude * diameter / (area * viscosity)
                models = self._friction_models[b]
                laminar = models.laminar(reynolds)
                turbulent = models.turbulent(reynolds)
                total = (
                    models.total(reynolds, laminar, turbulent)
                    * branch.friction_multiplier
                )
                resistance[b] = (
                    2.0
                    * total
                    * branch.length
                    * flow_magnitude
                    / (diameter * self.branch_density[b] * area ** 2)
                )
            elif branch.kind is NetworkBranchKind.VALVE:
                resistance[b] = (
                    branch.linear_resistance
                    + branch.quadratic_resistance * flow_magnitude
                )
            elif branch.kind is NetworkBranchKind.PUMP:
                characteristic = branch.characteristic
                resistance[b] = (
                    characteristic.linear_coefficient
                    + characteristic.quadratic_coefficient * flow_magnitude
                )
                source[b] = characteristic.head_at_zero_flow
        return resistance, source

    def _assemble_raw(self) -> tuple:
        """Mass diagonal, stiffness matrix and source vector of
        M dx/dt + K x = s at the frozen state."""
        number_of_internal = self.internal_node_indices.size
        mass = np.zeros(self.number_of_unknowns)
        stiffness = np.zeros(
            (self.number_of_unknowns, self.number_of_unknowns)
        )
        source = np.zeros(self.number_of_unknowns)
        resistance, branch_source = self._branch_resistance_and_source()

        # Node rows: C_i dp_i/dt + sum_b N[i, b] mdot_b = q_i.
        internal_incidence = self.incidence[self.internal_node_indices, :]
        mass[:number_of_internal] = self.node_capacitance[
            self.internal_node_indices
        ]
        stiffness[:number_of_internal, number_of_internal:] = (
            internal_incidence
        )
        for local, node_index in enumerate(self.internal_node_indices):
            source[local] = self.external_mass_sources.get(
                self.node_inputs[node_index].identifier, 0.0
            )

        # Branch rows: L_b dmdot_b/dt + R_b mdot_b - sum_i N[i, b] p_i
        # = dp_source_b; the reservoir part of the pressure difference is
        # known and moves to the source.
        branch_rows = number_of_internal + np.arange(len(self.branch_inputs))
        mass[number_of_internal:] = self.branch_inertance
        stiffness[branch_rows, branch_rows] = resistance
        stiffness[number_of_internal:, :number_of_internal] = (
            -internal_incidence.T
        )
        source[number_of_internal:] = branch_source + (
            self.incidence[self.reservoir_node_indices, :].T
            @ self.node_pressure[self.reservoir_node_indices]
        )
        return mass, stiffness, source

    def assemble_steady(self) -> tuple:
        """Stiffness matrix and source of the steady system K x = s."""
        _, stiffness, source = self._assemble_raw()
        return stiffness, source

    def assemble_transient(self, time_step: float) -> tuple:
        """System matrix and known-term vector of one time step, with the
        conductor's discretization semantics: same frozen matrices on both
        sides of the theta blend, fully implicit BDF2.

        Algebraic rows (no storage term) are enforced fully implicitly: the
        theta blend on a constraint row would only damp -- for
        Crank-Nicolson, indefinitely sustain -- an initial constraint
        residual instead of eliminating it. Node rows flagged in
        coupled_node_identifiers get the same treatment (see the attribute
        comment). BDF2 rows are implicit by construction."""
        mass, stiffness, source = self._assemble_raw()
        old_solution = self._solution_history[:, 0]
        if self.method is MethodFlag.BACKWARD_DIFFERENCE_2:
            a0, a1, a2 = self._bdf2_coefficients(time_step)
            matrix = a0 / time_step * np.diag(mass) + stiffness
            known_term = (
                mass
                / time_step
                * (a1 * old_solution - a2 * self._solution_history[:, 1])
                + source
            )
        else:
            previous_source = (
                source if self._previous_source is None
                else self._previous_source
            )
            matrix = np.diag(mass) / time_step + self.theta * stiffness
            known_term = (
                mass / time_step * old_solution
                - (1.0 - self.theta) * (stiffness @ old_solution)
                + self.theta * source
                + (1.0 - self.theta) * previous_source
            )
            implicit = mass == 0.0
            for identifier in self.coupled_node_identifiers:
                implicit[self.internal_node_unknown_index(identifier)] = True
            # Backward Euler on the flagged rows; for the algebraic rows
            # (zero mass) this reduces to the plain implicit constraint.
            matrix[implicit, :] = (
                np.diag(mass) / time_step + stiffness
            )[implicit, :]
            known_term[implicit] = (
                mass / time_step * old_solution + source
            )[implicit]
        self._last_source = source
        return matrix, known_term

    def _bdf2_coefficients(self, time_step: float) -> tuple:
        """Variable-step BDF2 coefficients (a0, a1, a2); mirrors
        step_matrix_construction.backward_difference_2_coefficients (which
        reads them from a Conductor), including the backward Euler start."""
        if self.num_step < 1 or not self.previous_time_step:
            return 1.0, 1.0, 0.0
        step_ratio = time_step / self.previous_time_step
        return (
            (1.0 + 2.0 * step_ratio) / (1.0 + step_ratio),
            1.0 + step_ratio,
            step_ratio ** 2 / (1.0 + step_ratio),
        )

    def _relative_change(
        self, solution: np.ndarray, state: np.ndarray
    ) -> float:
        """Relative change between Picard iterates, measured per unknown
        block (internal pressures, branch flows) and worst block taken.

        A single global norm would be dominated by the pressure unknowns
        (~1e5 Pa) and could report convergence while the much smaller flow
        unknowns (often ~1e-3 kg/s) are still moving -- the same
        mixed-scale problem the per-field magnitude floors solve in the
        transient thermal-hydraulic error estimator."""
        number_of_internal = self.internal_node_indices.size
        worst = 0.0
        for block in (
            slice(0, number_of_internal),
            slice(number_of_internal, None),
        ):
            block_norm = np.linalg.norm(solution[block])
            if block_norm == 0.0:
                continue
            worst = max(
                worst,
                np.linalg.norm(solution[block] - state[block]) / block_norm,
            )
        return worst

    def _solve(self, matrix: np.ndarray, known_term: np.ndarray) -> np.ndarray:
        try:
            solution = np.linalg.solve(matrix, known_term)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Singular hydraulic network system: check that every part "
                "of the network reaches a reservoir and that no branch has "
                "both zero linearized resistance and zero inertance."
            ) from error
        if not np.all(np.isfinite(solution)):
            raise RuntimeError(
                "Non-finite hydraulic network solution at step "
                f"{self.num_step + 1}."
            )
        return solution

    def step(self, time_step: float) -> np.ndarray:
        """Advance the network by one time step: freeze the properties at
        the old state, assemble, one linear solve (the network twin of the
        operating_conditions_th + step sequence of the conductor)."""
        self.update_properties()
        matrix, known_term = self.assemble_transient(time_step)
        solution = self._solve(matrix, known_term)
        self.advance(solution, time_step)
        return solution

    def advance(self, solution: np.ndarray, time_step: float):
        """Store the solution of a completed step and shift the history."""
        self._solution_history[:, 1] = self._solution_history[:, 0]
        self._solution_history[:, 0] = solution
        self._unpack_state(solution)
        self._previous_source = self._last_source
        self.previous_time_step = time_step
        self.num_step += 1
        # Fail fast on unphysical pressures, mirroring the post-solve field
        # check of the 1D solver: CoolProp would otherwise fail one property
        # call later with a misleading message.
        internal_pressures = self.node_pressure[self.internal_node_indices]
        if np.any(internal_pressures <= 0.0):
            bad = int(
                self.internal_node_indices[
                    np.argmin(internal_pressures)
                ]
            )
            raise RuntimeError(
                "Non-positive pressure at network node "
                f"{self.node_inputs[bad].identifier!r} after step "
                f"{self.num_step}: {self.node_pressure[bad]} Pa."
            )

    def initialize_branch_flows_from_pressures(
        self,
        max_iterations: int = 200,
        tolerance: float = 1.0e-12,
        relaxation: float = 0.5,
    ):
        """Initialize every branch flow from the user-given node pressures
        through the branch's own characteristic, leaving the pressures
        untouched.

        This respects the declared initial condition: any residual node
        imbalance is resolved by the transient itself -- through the
        compliance of nodes with volume, or instantaneously by the
        algebraic balance of zero-volume junctions, which moves flows, not
        pressures. (An initialization that relocated the node pressures to
        the network's own operating point would apply a pressure step to
        the coupled channel ends on the first time step.)

        Branches whose linearized resistance vanishes together with their
        driving pressure difference (e.g. an ideal pump exactly matched by
        its node pressures) keep their declared initial flow; the first
        transient step determines the flow through the node balances."""
        for b in range(len(self.branch_inputs)):
            if self.branch_mass_flow[b] == 0.0:
                self.branch_mass_flow[b] = STEADY_STATE_FLOW_SEED
        for _ in range(max_iterations):
            self.update_properties()
            resistance, pump_head = self._branch_resistance_and_source()
            driving = pump_head + (
                self.node_pressure[self._branch_from]
                - self.node_pressure[self._branch_to]
            )
            flows = np.where(
                resistance > 0.0,
                driving / np.where(resistance > 0.0, resistance, 1.0),
                np.array([
                    branch.initial_mass_flow for branch in self.branch_inputs
                ]),
            )
            change = np.abs(flows - self.branch_mass_flow).max() / max(
                np.abs(flows).max(), 1.0e-30
            )
            if change < tolerance:
                self.branch_mass_flow = flows
                break
            self.branch_mass_flow = (
                relaxation * flows
                + (1.0 - relaxation) * self.branch_mass_flow
            )
        self._solution_history[:, 0] = self._pack_state()
        self._solution_history[:, 1] = self._solution_history[:, 0]
        self._last_source = None
        self._previous_source = None
        self.num_step = 0
        self.previous_time_step = 0.0

    def solve_steady_state(
        self,
        max_iterations: int = 200,
        tolerance: float = 1.0e-10,
        relaxation: float = 0.5,
    ) -> np.ndarray:
        """Solve the steady network with a relaxed Picard iteration on the
        flow-linearized system; the network analog of the gen_flow
        initialization of the channels.

        The default relaxation 0.5 makes the fixed-point map of a purely
        quadratic resistance locally superlinear (the unrelaxed map has
        derivative -1 at the fixed point). Branches starting at exactly
        zero flow are seeded with a small flow so that quadratic-only
        resistances do not linearize to a singular first iteration.

        Returns the converged solution vector and resets the time
        integration history to the steady state."""
        for b in range(len(self.branch_inputs)):
            if self.branch_mass_flow[b] == 0.0:
                self.branch_mass_flow[b] = STEADY_STATE_FLOW_SEED
        state = self._pack_state()
        for _ in range(max_iterations):
            self.update_properties()
            stiffness, source = self.assemble_steady()
            solution = self._solve(stiffness, source)
            change = self._relative_change(solution, state)
            if change < tolerance:
                # Adopt the last unrelaxed solve: linearized at a converged
                # state it is the fixed point itself (exact for linear
                # networks), while the relaxed iterate still carries
                # O(tolerance) of the relaxation path -- which matters for
                # the flow unknowns when the convergence norm is dominated
                # by the much larger pressure unknowns.
                self._unpack_state(solution)
                break
            state = relaxation * solution + (1.0 - relaxation) * state
            self._unpack_state(state)
        else:
            raise RuntimeError(
                "Hydraulic network steady state did not converge within "
                f"{max_iterations} Picard iterations (last relative change "
                f"{change:.3e})."
            )
        steady_state = self._pack_state()
        self._solution_history[:, 0] = steady_state
        self._solution_history[:, 1] = steady_state
        self._last_source = None
        self._previous_source = None
        self.num_step = 0
        self.previous_time_step = 0.0
        return steady_state
