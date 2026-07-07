"""Lumped-parameter hydraulic network (hydraulic field-circuit coupling).

The network is defined by the ``hydraulic_network:`` section of
``simulation.yaml`` (parsed and validated in
:mod:`hydraulics.network.network_inputs`) and solved by
:class:`hydraulics.network.hydraulic_network.HydraulicNetwork` -- standalone
(stage 0) or coupled to conductor channel ends through the port
declarations, resolved and solved monolithically with the bordered Schur
complement machinery of :mod:`hydraulics.network.coupling` (stage 1).
"""

from hydraulics.network.network_inputs import (
    HydraulicNetworkInput,
    NetworkBranchInput,
    NetworkBranchKind,
    NetworkNodeInput,
    NetworkNodeKind,
    PortEnd,
    PortInput,
    PumpCharacteristic,
)
from hydraulics.network.hydraulic_network import HydraulicNetwork
from hydraulics.network.coupling import (
    ResolvedPort,
    apply_network_port_boundary_conditions,
    build_coupled_network,
    resolve_network_coupling,
    solve_coupled_step,
)
