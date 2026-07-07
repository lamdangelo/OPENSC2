"""Lumped-parameter hydraulic network (hydraulic field-circuit coupling).

Stage 0 of the coupling plan: the network is defined by the
``hydraulic_network:`` section of ``simulation.yaml`` (parsed and validated
in :mod:`hydraulics.network.network_inputs`) and solved standalone by
:class:`hydraulics.network.hydraulic_network.HydraulicNetwork`. The port
declarations that couple network nodes to conductor channel ends are parsed
and validated against the network topology here, but their resolution to
FluidComponent boundary conditions (the bordered solve) is stage 1.
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
