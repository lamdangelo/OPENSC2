"""V1 — Steady Kirchhoff check on a Wheatstone bridge of laminar pipes.

A supply and a return reservoir feed a Wheatstone bridge of five pipe
branches with distinct laminar (Hagen-Poiseuille) resistances through two
connection pipes. The production steady solve
(HydraulicNetwork.solve_steady_state, a relaxed Picard iteration whose final
unrelaxed solve is exact for a linear network) is compared against the
hand-assembled reduced nodal system A_int G A_int^T p = b solved with numpy
(analytical.steady_network_reference).

The lengths are chosen so the bridge is unbalanced (R1/R3 != R2/R5), keeping
a finite flow in the bridge branch, and so every branch Reynolds number stays
inside the exactly-laminar band of the friction pairing (Re < ~1187, where
total = max(laminar, turbulent) provably returns f = 16/Re and the branch
resistance is exactly R = 32 mu l / (D^2 rho A), independent of the
LAMINAR_REYNOLDS_FLOOR).
"""

import numpy as np
import pytest

import analytical
from case_builders import (
    LAMINAR_REYNOLDS_LIMIT,
    VERIFICATION_FLUID,
    build_network,
    internal,
    pipe,
    reservoir,
)

SUPPLY_PRESSURE = 1.05e5  # Pa
RETURN_PRESSURE = 1.00e5  # Pa
DIAMETER = 2.0e-3  # m
CROSS_SECTION = np.pi / 4.0 * DIAMETER**2  # m^2

# identifier -> (from_node, to_node, length [m]); "supply"/"sink" are the
# reservoirs, "inlet"/"outlet"/"left"/"right" the four zero-volume junctions.
BRANCHES = {
    "feed": ("supply", "inlet", 0.5),
    "bridge_1": ("inlet", "left", 1.0),
    "bridge_2": ("inlet", "right", 2.0),
    "bridge_3": ("left", "outlet", 3.0),
    "bridge_4": ("right", "outlet", 4.0),
    "bridge_5": ("left", "right", 1.5),
    "drain": ("outlet", "sink", 0.8),
}
INTERNAL_NODES = ["inlet", "left", "right", "outlet"]
RESERVOIR_NODES = ["supply", "sink"]


def wheatstone_network():
    nodes = [
        reservoir("supply", SUPPLY_PRESSURE),
        reservoir("sink", RETURN_PRESSURE),
    ] + [
        # Initial guesses between the reservoir pressures; the steady solve
        # replaces them.
        internal(identifier, 1.02e5)
        for identifier in INTERNAL_NODES
    ]
    branches = [
        pipe(identifier, from_node, to_node, length, DIAMETER, CROSS_SECTION)
        for identifier, (from_node, to_node, length) in BRANCHES.items()
    ]
    return build_network(nodes, branches)


def incidence_matrices():
    """Oriented incidence (+1 leaving / -1 entering), split into internal and
    reservoir rows, in the local orderings INTERNAL_NODES / RESERVOIR_NODES /
    BRANCHES."""
    internal_incidence = np.zeros((len(INTERNAL_NODES), len(BRANCHES)))
    reservoir_incidence = np.zeros((len(RESERVOIR_NODES), len(BRANCHES)))
    for column, (from_node, to_node, _) in enumerate(BRANCHES.values()):
        for node, sign in ((from_node, +1.0), (to_node, -1.0)):
            if node in INTERNAL_NODES:
                internal_incidence[INTERNAL_NODES.index(node), column] = sign
            else:
                reservoir_incidence[RESERVOIR_NODES.index(node), column] = sign
    return internal_incidence, reservoir_incidence


def test_wheatstone_bridge_matches_kirchhoff_reference(constant_fluid):
    network = wheatstone_network()
    network.solve_steady_state()

    resistances = np.array(
        [
            analytical.laminar_pipe_resistance(
                constant_fluid.viscosity,
                length,
                DIAMETER,
                constant_fluid.density,
                CROSS_SECTION,
            )
            for (_, _, length) in BRANCHES.values()
        ]
    )
    internal_incidence, reservoir_incidence = incidence_matrices()
    reference_pressures, reference_flows = analytical.steady_network_reference(
        internal_incidence,
        reservoir_incidence,
        resistances,
        np.array([SUPPLY_PRESSURE, RETURN_PRESSURE]),
    )

    solver_pressures = np.array(
        [network.node_pressure_of(identifier) for identifier in INTERNAL_NODES]
    )
    solver_flows = np.array(
        [network.branch_mass_flow_of(identifier) for identifier in BRANCHES]
    )

    # The problem is exactly linear (laminar resistances are flow-independent)
    # and both sides are float64 linear solves, so agreement is limited only
    # by rounding (~1e-15 relative); 1e-8 verifies that with a wide margin
    # while also certifying the exact Hagen-Poiseuille reduction of the pipe
    # law. Flows get an absolute floor scaled to the largest flow so the
    # small bridge-branch flow is compared on the system scale.
    np.testing.assert_allclose(solver_pressures, reference_pressures, rtol=1.0e-8)
    largest_flow = np.max(np.abs(reference_flows))
    np.testing.assert_allclose(
        solver_flows, reference_flows, rtol=1.0e-8, atol=1.0e-8 * largest_flow
    )

    # Exact mass conservation of the returned solution: the net flow at every
    # internal node must vanish to solver round-off (1e-12 of the flow scale).
    node_balance = internal_incidence @ solver_flows
    assert np.max(np.abs(node_balance)) < 1.0e-12 * largest_flow

    # Sanity: every branch operates inside the exactly-laminar band, so the
    # analytical resistances above are provably the ones the solver used.
    reynolds = (
        np.abs(solver_flows)
        * DIAMETER
        / (CROSS_SECTION * constant_fluid.viscosity)
    )
    assert np.all(reynolds < LAMINAR_REYNOLDS_LIMIT)
    # The bridge branch must carry finite flow (unbalanced bridge), otherwise
    # the flow comparison would be trivial there.
    bridge_flow = solver_flows[list(BRANCHES).index("bridge_5")]
    assert abs(bridge_flow) > 1.0e-3 * largest_flow
