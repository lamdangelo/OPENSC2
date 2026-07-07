"""Hydraulic field-circuit coupling: conductor channel ends coupled to
lumped-network nodes through a monolithic bordered (Schur complement)
solve around the banded thermal-hydraulic systems.

Every coupled port is pressure-driven, mirroring the imposed-pressure
boundary condition it replaces: after the standard boundary conditions are
applied, :func:`apply_network_port_boundary_conditions` rewrites the
port-end pressure row into the continuity constraint

    p_port - p_node = 0

whose identity part stays in the banded matrix while the -1 on the network
node pressure lands in a border column. The port mass flow

    mdot_into_node = sigma * rho_old * A * v_port,   sigma = +1 at x = L,
                                                             -1 at x = 0

enters the node mass balance implicitly through the coupling block (rho
frozen at the old state, so the flow is linear in the unknown port
velocity -- the same Picard linearization as everywhere else). The port
flow is the Lagrange multiplier enforcing pressure continuity, exactly as
terminal currents are in field-circuit coupling for electromagnetics.

The coupled linear system (one banded block A_k per coupled conductor)

    [ A_1        B_1 ] [x_1  ]   [b_1  ]
    [      A_2   B_2 ] [x_2  ] = [b_2  ]
    [ C_1  C_2    D  ] [x_net]   [f_net]

is solved by bordering: one banded solve per conductor with the stacked
right-hand sides [b_k | B_k] (each A_k is nonsingular on its own, the
continuity rows keep their unit diagonal), then one small dense Schur
system (D - sum_k C_k A_k^-1 B_k) x_net = f_net - sum_k C_k A_k^-1 b_k
closing all conductors and the network together, then back-substitution
per conductor. The banded structures, the row scaling and the
one-linear-solve-per-step architecture of the field solver all survive
untouched; conductors coupled to one network meet only in the Schur
complement, and must therefore share the time step and the integration
method (enforced by the simulation loop and by the network build).

Restriction enforced at resolution time: each port end must be one where
the channel's own boundary condition imposes pressure (the port takes over
exactly that row).
"""

import warnings
from dataclasses import dataclass

import numpy as np

from hydraulics.hydraulic_flags import FlowDirection, HydraulicBC
from hydraulics.network.hydraulic_network import HydraulicNetwork
from hydraulics.network.network_inputs import (
    HydraulicNetworkInput,
    PortEnd,
)

# Channel boundary conditions that impose pressure at a given end: a port
# may only replace a pressure row.
_PRESSURE_IMPOSING_BC = {
    PortEnd.INLET: (
        HydraulicBC.IMPOSE_PRESSURE_DROP,
        HydraulicBC.IMPOSE_INLET_PRESSURE_OUTLET_VELOCITY,
    ),
    PortEnd.OUTLET: (
        HydraulicBC.IMPOSE_PRESSURE_DROP,
        HydraulicBC.IMPOSE_INLET_VELOCITY_OUTLET_PRESSURE,
    ),
}


@dataclass
class ResolvedPort:
    """A network port resolved to a fluid component end.

    The mesh end and the equation indices follow the channel's
    flow-direction convention through the inl_idx/out_idx namedtuples built
    by FluidComponent.build_th_bc_index; the (possibly negative) indices
    address the assembled vectors directly and are normalized with
    ``% total_equations`` where an absolute position is needed."""
    fluid_component: object
    end: PortEnd
    network: HydraulicNetwork
    node_identifier: str
    node_unknown_index: int
    mesh_end_is_last_node: bool
    pressure_index: int
    velocity_index: int
    temperature_index: int

    @property
    def flow_orientation_sign(self) -> float:
        """+1 if the channel axis points out of the channel at the port
        (x = L end), -1 otherwise (x = 0 end): the sign turning rho*A*v
        into the mass flow entering the network node."""
        return 1.0 if self.mesh_end_is_last_node else -1.0

    @property
    def end_node_slice(self) -> int:
        """Index of the port's mesh node in a nodal field array."""
        return -1 if self.mesh_end_is_last_node else 0

    def mass_flow_into_node(self) -> float:
        """Port mass flow entering the network node at the current state."""
        fields = self.fluid_component.coolant.node_fields
        return (
            self.flow_orientation_sign
            * fields.total_density[self.end_node_slice]
            * self.fluid_component.channel.inputs.cross_section
            * fields.velocity[self.end_node_slice]
        )


def _resolve_port_indices(fluid_component, end: PortEnd) -> tuple:
    """(pressure, velocity, temperature) equation indices of the channel
    end a port couples to."""
    forward = (
        fluid_component.coolant.operations.flow_direction
        is FlowDirection.FORWARD
    )
    bc_idx = (
        fluid_component.inl_idx if end is PortEnd.INLET
        else fluid_component.out_idx
    )
    return tuple(
        pair.forward if forward else pair.backward
        for pair in (bc_idx.pressure, bc_idx.velocity, bc_idx.temperature)
    )


def resolve_network_coupling(network: HydraulicNetwork, conductors: list):
    """Resolve the port declarations of the network input against the
    instantiated conductors.

    Sets ``conductor.network_ports`` on the coupled conductor and flags the
    coupled nodes in ``network.coupled_node_identifiers``. Must run after
    conductor initialization (the ports reuse the boundary-condition
    equation indices built there)."""
    conductor_by_identifier = {
        conductor.identifier: conductor for conductor in conductors
    }
    for port_input in network.inputs.ports:
        conductor = conductor_by_identifier.get(port_input.conductor)
        if conductor is None:
            raise ValueError(
                f"Network port references unknown conductor "
                f"{port_input.conductor!r}."
            )
        fluid_component = next(
            (
                f_comp
                for f_comp in conductor.inventory.fluids.collection
                if f_comp.identifier == port_input.channel
            ),
            None,
        )
        if fluid_component is None:
            raise ValueError(
                f"Network port references unknown channel "
                f"{port_input.channel!r} of conductor "
                f"{port_input.conductor!r}."
            )
        if fluid_component.coolant.fluid_type is not network.fluid_type:
            raise ValueError(
                f"Fluid type mismatch at network port {port_input.node!r}: "
                f"network carries {network.fluid_type.value}, channel "
                f"{port_input.channel!r} carries "
                f"{fluid_component.coolant.fluid_type.value}."
            )
        bc_type = fluid_component.coolant.operations.hydraulic_bc_type
        if bc_type not in _PRESSURE_IMPOSING_BC[port_input.end]:
            raise ValueError(
                f"Network port at the {port_input.end.value} of "
                f"{port_input.channel!r} requires a boundary condition that "
                f"imposes pressure at that end (INTIAL 1, or 2/3 with the "
                f"pressure at the port end); found {bc_type.name}."
            )
        pressure_index, velocity_index, temperature_index = (
            _resolve_port_indices(fluid_component, port_input.end)
        )
        forward = (
            fluid_component.coolant.operations.flow_direction
            is FlowDirection.FORWARD
        )
        conductor.network_ports.append(
            ResolvedPort(
                fluid_component=fluid_component,
                end=port_input.end,
                network=network,
                node_identifier=port_input.node,
                node_unknown_index=network.internal_node_unknown_index(
                    port_input.node
                ),
                mesh_end_is_last_node=(
                    (port_input.end is PortEnd.OUTLET) == forward
                ),
                pressure_index=pressure_index,
                velocity_index=velocity_index,
                temperature_index=temperature_index,
            )
        )
        network.coupled_node_identifiers.add(port_input.node)


def warn_on_partially_ported_parallel_groups(network: HydraulicNetwork,
                                             conductor):
    """Warn when a ported channel has hydraulic-parallel partners (open
    interface along the length) that are not ported to the same node at the
    same end.

    A network node then moves the ported channel's end pressure relative to
    its partners' fixed boundary values, and a differential plenum pressure
    between openly connected parallel channels is strongly amplified by the
    transverse exchange terms (measured on CASE_1: a common-mode 10 Pa
    boundary shift responds with ~1e-5 K, the same shift on one channel
    only with ~1 K scale). Legitimate only when the differential stays
    negligible."""
    ported = {
        (port.fluid_component.identifier, port.end): port.node_identifier
        for port in conductor.network_ports
    }
    groups = conductor.dict_topology["ch_ch"]["Hydraulic_parallel"]
    for group_data in groups.values():
        group = [f_comp.identifier for f_comp in group_data["Group"]]
        for (channel, end), node in ported.items():
            if channel not in group:
                continue
            unmatched = [
                partner for partner in group
                if partner != channel and ported.get((partner, end)) != node
            ]
            if unmatched:
                warnings.warn(
                    f"Channel {channel!r} is ported to network node "
                    f"{node!r} at its {end.value}, but its hydraulic-"
                    f"parallel partner(s) {unmatched} are not ported to the "
                    "same node there. The network can then drive a "
                    "differential plenum pressure between openly connected "
                    "channels, which is strongly amplified by the "
                    "transverse exchange terms; port all channels of the "
                    "group to the shared manifold unless the differential "
                    "is guaranteed to stay negligible."
                )


def build_coupled_network(simulation, mapping: dict) -> HydraulicNetwork:
    """Build, resolve and initialize the hydraulic network declared in the
    ``hydraulic_network:`` section of simulation.yaml.

    The network inherits the (common, validated) time integration method
    of the conductors it couples to, so the coupled step has one
    consistent discretization."""
    inputs = HydraulicNetworkInput.from_mapping(mapping)
    if not inputs.ports:
        raise ValueError(
            "hydraulic_network declared without ports: the network would "
            "never be advanced. Declare at least one port or remove the "
            "section."
        )
    coupled_identifiers = {port.conductor for port in inputs.ports}
    coupled_conductors = [
        conductor
        for conductor in simulation.list_of_Conductors
        if conductor.identifier in coupled_identifiers
    ]
    missing = coupled_identifiers - {
        conductor.identifier for conductor in coupled_conductors
    }
    if missing:
        raise ValueError(
            f"Network port(s) reference unknown conductor(s) "
            f"{sorted(missing)}."
        )
    # The coupled step has one consistent time discretization: all coupled
    # conductors and the network share the integration method.
    methods = {
        conductor.inputs.thermohydraulic_method
        for conductor in coupled_conductors
    }
    if len(methods) > 1:
        raise ValueError(
            "All conductors coupled to the hydraulic network must share "
            "the same thermohydraulic method; found "
            f"{sorted(method.name for method in methods)}."
        )
    network = HydraulicNetwork(inputs, method=methods.pop())
    resolve_network_coupling(network, simulation.list_of_Conductors)
    for conductor in coupled_conductors:
        warn_on_partially_ported_parallel_groups(network, conductor)
    # The network state time evolution is written once per time step, into
    # the Time_evolution directory of the first coupled conductor.
    network.output_conductor_identifier = coupled_conductors[0].identifier
    # Initialize the branch flows from the declared node pressures without
    # relocating the pressures themselves: the coupled channels were
    # initialized against those values, and moving a port node to the
    # network's own operating point would step the channel end pressure on
    # the first time step.
    network.initialize_branch_flows_from_pressures()
    return network


def apply_network_port_boundary_conditions(
    conductor,
    known_term: np.ndarray,
    system_matrix: np.ndarray,
) -> tuple:
    """Overlay the port rows on the assembled system, after the standard
    boundary conditions of every channel have been applied.

    The port-end pressure row becomes the continuity constraint (identity
    on the port pressure, zero right-hand side; the network node column is
    added as a border by solve_coupled_step). The port-end temperature row
    is imposed from the network node temperature whenever the flow enters
    the channel there, with the same velocity-sign test as the standard
    temperature boundary conditions."""
    main_diagonal = conductor.band.number_of_subdiagonals
    for port in conductor.network_ports:
        system_matrix[:, port.pressure_index] = 0.0
        system_matrix[main_diagonal, port.pressure_index] = 1.0
        known_term[port.pressure_index] = 0.0

        end_velocity = port.fluid_component.coolant.node_fields.velocity[
            port.end_node_slice
        ]
        inflow = (
            end_velocity < 0.0 if port.mesh_end_is_last_node
            else end_velocity > 0.0
        )
        if inflow:
            system_matrix[:, port.temperature_index] = 0.0
            system_matrix[main_diagonal, port.temperature_index] = 1.0
            known_term[port.temperature_index] = (
                port.network.node_temperature[
                    port.network.node_index[port.node_identifier]
                ]
            )
    return known_term, system_matrix


def _coupling_blocks(conductor, network: HydraulicNetwork,
                     row_scaling_factors: np.ndarray) -> tuple:
    """Border columns B (field rows x network unknowns) and coupling block
    C (network rows x field unknowns) of one conductor's ports."""
    number_of_equations = conductor.equation_counts.total_equations
    border = np.zeros((number_of_equations, network.number_of_unknowns))
    coupling = np.zeros((network.number_of_unknowns, number_of_equations))
    for port in conductor.network_ports:
        pressure_row = port.pressure_index % number_of_equations
        velocity_column = port.velocity_index % number_of_equations
        # Continuity row: p_port - p_node = 0.
        border[pressure_row, port.node_unknown_index] = -1.0
        # Node mass balance: ... - mdot_into_node = q, with
        # mdot_into_node = sigma * rho_old * A * v_port.
        coupling[port.node_unknown_index, velocity_column] += (
            -port.flow_orientation_sign
            * port.fluid_component.coolant.node_fields.total_density[
                port.end_node_slice
            ]
            * port.fluid_component.channel.inputs.cross_section
        )
    # The field rows were equilibrated after the boundary conditions; the
    # border columns belong to those rows and scale identically.
    border /= row_scaling_factors[:, np.newaxis]
    return border, coupling


def solve_coupled_conductors_step(conductors: list,
                                  network: HydraulicNetwork,
                                  qsource_by_identifier: dict):
    """One monolithic time step of every network-coupled conductor and the
    network (see module docstring).

    Each conductor is assembled with the standard phase functions, solved
    against the stacked right-hand sides [b_k | B_k] with its own banded
    factorization, and contributes C_k A_k^-1 B_k to the shared dense Schur
    complement of the network unknowns; one small dense solve then closes
    all conductors and the network simultaneously, and the network state is
    advanced exactly once. The simulation loop must have harmonized the
    time step of the coupled conductors beforehand.

    The network is solved for the increment delta = x_net - x_net_old
    rather than x_net itself: with the raw continuity right-hand side
    (zero) the particular field solutions would correspond to zero port
    pressure -- a state far from the final one -- and recovering the answer
    would subtract two large near-cancelling vectors, losing several
    digits. Shifted by the old network state, the particular solves run at
    p_port = p_node_old (essentially the true state) and the border
    corrections stay small."""
    # Deferred import: transient_solution_functions imports this module.
    from utility_functions.transient_solution_functions import (
        assemble_thermal_hydraulic_system,
        finalize_step,
        solve_thermal_banded_system,
    )

    time_steps = {conductor.time_step for conductor in conductors}
    if len(time_steps) > 1:
        raise ValueError(
            "Network-coupled conductors must share the time step; got "
            f"{sorted(time_steps)}."
        )
    time_step = conductors[0].time_step

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
        border, coupling = _coupling_blocks(
            conductor, network, row_scaling_factors
        )
        stacked_solution = solve_thermal_banded_system(
            conductor,
            system_matrix,
            np.column_stack(
                (known_term - border @ old_network_state, border)
            ),
        )
        particular = stacked_solution[:, 0]
        border_influence = stacked_solution[:, 1:]
        schur_matrix -= coupling @ border_influence
        schur_known -= coupling @ particular
        conductor_solves.append(
            (conductor, known_term, row_scaling_factors, particular,
             border_influence)
        )

    network_increment = np.linalg.solve(schur_matrix, schur_known)

    for (conductor, known_term, row_scaling_factors, particular,
         border_influence) in conductor_solves:
        finalize_step(
            conductor,
            particular - border_influence @ network_increment,
            known_term,
            row_scaling_factors,
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
