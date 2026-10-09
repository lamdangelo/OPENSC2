"""One transient step of the conductors coupled to a lumped hydraulic
network: bordered (Schur) field + network solve.

Lives with the stepper (numerics) rather than with the network coupling
(hydraulics.network.coupling): it orchestrates assemble -> solve ->
finalize of several conductors around the network advance, so it depends
on both, and neither of them depends on it.
"""

from dataclasses import dataclass

import numpy as np

from opensc2.hydraulics.hydraulic_flags import MASS_FLOW_FORMULATIONS
from opensc2.hydraulics.network.hydraulic_network import HydraulicNetwork
from opensc2.numerics.transient_solution_functions import (
    assemble_thermal_hydraulic_system,
    finalize_step,
    solve_thermal_banded_system,
)
from opensc2.ports.diagnostics import DiagnosticsSink, NULL_DIAGNOSTICS


@dataclass
class _CouplingStructure:
    """Sparse form of one conductor's border B and coupling block C.

    B (field rows x network unknowns) is nonzero only in the columns of the
    nodes this conductor is ported to -- typically two -- so only those
    columns are stored and solved for; the rest are exact zeros. C (network
    rows x field unknowns) is nonzero only at the ported node rows and, in
    each, only at the port velocity columns, so it is stored as coefficient
    triplets rather than a dense (network x field) array.

    The dropped columns/rows are exact zeros, so the reduced products equal
    the dense ones up to BLAS summation-order (last-bit) reordering -- far
    below the roundoff already inherent in the stacked-right-hand-side path.

    Layout:
        * ``ported_columns``: the network unknown indices B touches, sorted
          ascending so the reduced sum runs over the same nonzero columns in
          the same order the full B @ x would.
        * ``border_reduced``: (field rows x len(ported_columns)) B restricted
          to those columns, already row-scaled.
        * ``coupling_nodes`` / ``coupling_velocity_columns`` /
          ``coupling_coefficients``: one triplet per port,
          C[node, velocity_column] += coefficient. Ordered by velocity
          column so the scatter-add accumulates each node's contributions in
          the same ascending-column order a dense C @ x would.
    """
    ported_columns: np.ndarray
    border_reduced: np.ndarray
    coupling_nodes: np.ndarray
    coupling_velocity_columns: np.ndarray
    coupling_coefficients: np.ndarray
    number_of_network_unknowns: int

    def coupling_times(self, vectors: np.ndarray) -> np.ndarray:
        """C @ vectors, where ``vectors`` is the field solution (shape
        (n_eq,)) or several field columns (shape (n_eq, k)); returns a
        length-(network unknowns) vector or (network unknowns, k) block with
        the port contributions scattered onto the ported node rows."""
        contributions = (
            self.coupling_coefficients.reshape(
                (-1,) + (1,) * (vectors.ndim - 1)
            )
            * vectors[self.coupling_velocity_columns]
        )
        result = np.zeros(
            (self.number_of_network_unknowns,) + vectors.shape[1:]
        )
        np.add.at(result, self.coupling_nodes, contributions)
        return result


def _coupling_blocks(conductor, network: HydraulicNetwork,
                     row_scaling_factors: np.ndarray) -> _CouplingStructure:
    """Sparse border B and coupling block C of one conductor's ports (see
    :class:`_CouplingStructure`).

    B carries the -1 on the network node pressure of each port's continuity
    row; C carries -sigma * rho_old * A on each port's velocity column,
    feeding mdot_into_node = sigma * rho_old * A * v_port into the node mass
    balance (rho frozen at the old state, so the flow is linear in the
    unknown port velocity -- the same Picard linearization as everywhere
    else)."""
    number_of_equations = conductor.equation_counts.total_equations
    ports = conductor.network_ports
    ported_columns = np.unique(
        [port.node_unknown_index for port in ports]
    ).astype(int)
    local_of_node = {
        int(column): local for local, column in enumerate(ported_columns)
    }
    border_reduced = np.zeros((number_of_equations, ported_columns.size))
    coupling_nodes = np.empty(len(ports), dtype=int)
    coupling_velocity_columns = np.empty(len(ports), dtype=int)
    coupling_coefficients = np.empty(len(ports))
    for k, port in enumerate(ports):
        pressure_row = port.pressure_index % number_of_equations
        velocity_column = port.velocity_index % number_of_equations
        # Continuity row: p_port - p_node = 0.
        border_reduced[pressure_row, local_of_node[port.node_unknown_index]] = (
            -1.0
        )
        coupling_nodes[k] = port.node_unknown_index
        coupling_velocity_columns[k] = velocity_column
        # Node mass balance: ... - mdot_into_node = q.
        if (
            port.fluid_component.coolant.hydraulic_formulation
            in MASS_FLOW_FORMULATIONS
        ):
            # The port slot is the native mass flow rate: the coupling
            # coefficient is exact (+-1), with no frozen-density Picard
            # linearization left on this block.
            coupling_coefficients[k] = -port.flow_orientation_sign
        else:
            coupling_coefficients[k] = (
                -port.flow_orientation_sign
                * port.fluid_component.coolant.node_fields.total_density[
                    port.end_node_slice
                ]
                * port.fluid_component.channel.inputs.cross_section
            )
    # The field rows were equilibrated after the boundary conditions; the
    # border columns belong to those rows and scale identically.
    border_reduced /= row_scaling_factors[:, np.newaxis]
    # Order the coupling triplets by velocity column so the scatter-add in
    # coupling_times accumulates each node's contributions in the same
    # ascending-column order a dense C @ x would.
    order = np.argsort(coupling_velocity_columns, kind="stable")
    return _CouplingStructure(
        ported_columns=ported_columns,
        border_reduced=border_reduced,
        coupling_nodes=coupling_nodes[order],
        coupling_velocity_columns=coupling_velocity_columns[order],
        coupling_coefficients=coupling_coefficients[order],
        number_of_network_unknowns=network.number_of_unknowns,
    )


def solve_coupled_conductors_step(conductors: list,
                                  network: HydraulicNetwork,
                                  qsource_by_identifier: dict,
                                  diagnostics: DiagnosticsSink = NULL_DIAGNOSTICS):
    """One monolithic time step of every network-coupled conductor and the
    network (see module docstring).

    Each conductor is assembled with the standard phase functions, solved
    against the stacked right-hand sides [b_k | B_k] with its own banded
    factorization (only the border columns of the nodes the conductor is
    ported to are carried; the rest of B_k is zero), and contributes
    C_k A_k^-1 B_k to the shared dense Schur complement of the network
    unknowns; one small dense solve then closes all conductors and the
    network simultaneously, and the network state is advanced exactly once.
    The simulation loop must have harmonized the time step of the coupled
    conductors beforehand.

    The network is solved for the increment delta = x_net - x_net_old
    rather than x_net itself: with the raw continuity right-hand side
    (zero) the particular field solutions would correspond to zero port
    pressure -- a state far from the final one -- and recovering the answer
    would subtract two large near-cancelling vectors, losing several
    digits. Shifted by the old network state, the particular solves run at
    p_port = p_node_old (essentially the true state) and the border
    corrections stay small."""
    time_steps = {conductor.time_state.time_step for conductor in conductors}
    if len(time_steps) > 1:
        raise ValueError(
            "Network-coupled conductors must share the time step; got "
            f"{sorted(time_steps)}."
        )
    time_step = conductors[0].time_state.time_step

    network.update_properties()
    network_matrix, network_known = network.assemble_transient(time_step)
    old_network_state = network.state_vector()

    schur_matrix = network_matrix.copy()
    schur_known = network_known - network_matrix @ old_network_state
    conductor_solves = []
    for conductor in conductors:
        system_matrix, known_term, row_scaling_factors = (
            assemble_thermal_hydraulic_system(
                conductor, qsource_by_identifier[conductor.identifier]
            )
        )
        blocks = _coupling_blocks(conductor, network, row_scaling_factors)
        ported_columns = blocks.ported_columns
        # Solve only against the border columns of the nodes this conductor
        # is ported to; the other columns of B are exact zeros and would
        # return zero solves. old_network_state is restricted to the same
        # columns for the increment shift of the particular right-hand side.
        stacked_solution = solve_thermal_banded_system(
            conductor,
            system_matrix,
            np.column_stack(
                (
                    known_term
                    - blocks.border_reduced @ old_network_state[ported_columns],
                    blocks.border_reduced,
                )
            ),
        )
        particular = stacked_solution[:, 0]
        border_influence = stacked_solution[:, 1:]
        # Schur accumulation, restricted to the ported node columns/rows: the
        # dropped entries are zero, so this reproduces the dense update.
        schur_matrix[:, ported_columns] -= blocks.coupling_times(
            border_influence
        )
        schur_known -= blocks.coupling_times(particular)
        conductor_solves.append(
            (conductor, known_term, row_scaling_factors, particular,
             border_influence, ported_columns)
        )

    network_increment = np.linalg.solve(schur_matrix, schur_known)

    for (conductor, known_term, row_scaling_factors, particular,
         border_influence, ported_columns) in conductor_solves:
        finalize_step(
            conductor,
            particular - border_influence @ network_increment[ported_columns],
            known_term,
            row_scaling_factors,
            diagnostics,
        )
    network.advance(old_network_state + network_increment, time_step)

    # Network-side energy transport: every port reports its new-state mass
    # flow and temperature to the node enthalpy balances (channel-feeding
    # ports carry the node's own temperature and drop out there).
    port_inflows = {}
    for conductor in conductors:
        for port in conductor.network_ports:
            port_inflows.setdefault(port.node_identifier, []).append(
                (
                    port.mass_flow_into_node(),
                    float(
                        port.fluid_component.coolant.node_fields.temperature[
                            port.end_node_slice
                        ]
                    ),
                )
            )
    network.update_node_temperatures(time_step, port_inflows)
