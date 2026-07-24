"""Input dataclasses and parser of the lumped-parameter hydraulic network.

The network is declared in the optional top-level ``hydraulic_network:``
section of ``simulation.yaml``; :meth:`HydraulicNetworkInput.from_mapping`
consumes the already-loaded YAML mapping. This is a new input section with
no legacy workbook counterpart, so the keys are born descriptive and no
schema-v2 alias table is involved.

Sign and orientation conventions (shared with
:mod:`hydraulics.network.hydraulic_network`):
    * a branch is oriented from ``from`` to ``to``; a positive mass flow
      rate runs in that direction;
    * a pump raises the pressure from ``from`` to ``to`` following its
      characteristic ``dp(mdot) = a0 - a1*mdot - a2*mdot*|mdot|``.

Node temperatures are prescribed inputs: network-side energy transport
(node enthalpy balances) is a later stage of the coupling plan.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from hydraulics.friction_factor_models import FrictionFactorModelType
from hydraulics.hydraulic_flags import FluidType, get_fluid_type


class NetworkNodeKind(Enum):
    # Pressure unknown, evolves with the node mass balance.
    INTERNAL = "internal"
    # Fixed pressure and temperature: the pressure reference of the network
    # (the ground / ideal-source analog of the MNA). Validation requires at
    # least one per connected part of the network.
    RESERVOIR = "reservoir"


def get_network_node_kind(flag: str) -> NetworkNodeKind:
    try:
        return NetworkNodeKind(str(flag).lower())
    except ValueError:
        raise ValueError(f"Unknown network node kind {flag!r}.")


class NetworkBranchKind(Enum):
    # Distributed friction from a friction-factor correlation (reuses the
    # channel friction models) plus the l/A inertance.
    PIPE = "pipe"
    # Lumped resistance dp = linear_resistance * mdot
    #                      + quadratic_resistance * mdot * |mdot|.
    VALVE = "valve"
    # Source branch with the characteristic of PumpCharacteristic.
    PUMP = "pump"
    # Pressure-triggered by-pass with hysteresis (quench protection):
    # blocked until the monitored node exceeds trip_pressure, then a
    # one-directional valve with the given open-state resistances until
    # the monitored pressure falls below reseat_pressure. State is
    # switched once per step on the previous-step pressures (as the
    # check valve switches on the previous-step flow sign).
    RELIEF_VALVE = "relief_valve"


def get_network_branch_kind(flag: str) -> NetworkBranchKind:
    try:
        return NetworkBranchKind(str(flag).lower())
    except ValueError:
        raise ValueError(f"Unknown network branch kind {flag!r}.")


class PortEnd(Enum):
    """Channel end a port couples to, in the inlet/outlet convention of the
    channel operations (resolved to a mesh end through flow_direction, as in
    FluidComponent.build_th_bc_index)."""
    INLET = "inlet"
    OUTLET = "outlet"


def get_port_end(flag: str) -> PortEnd:
    try:
        return PortEnd(str(flag).lower())
    except ValueError:
        raise ValueError(f"Unknown port end {flag!r}.")


@dataclass
class NetworkNodeInput:
    identifier: str
    kind: NetworkNodeKind
    # Pa; fixed value for a reservoir, initial value for an internal node.
    pressure: float
    # K; prescribed for both kinds (see module docstring).
    temperature: float
    # m^3; compliance volume of an internal node. 0 makes the node a pure
    # junction whose mass balance is an algebraic constraint.
    volume: float = 0.0


@dataclass
class PumpCharacteristic:
    """dp(mdot) = head_at_zero_flow - linear_coefficient * mdot
    - quadratic_coefficient * mdot * |mdot|."""
    head_at_zero_flow: float  # Pa
    linear_coefficient: float = 0.0  # Pa/(kg/s)
    quadratic_coefficient: float = 0.0  # Pa/(kg/s)^2


@dataclass
class NetworkBranchInput:
    identifier: str
    kind: NetworkBranchKind
    from_node: str
    to_node: str

    # PIPE geometry and friction correlation.
    length: float = 0.0  # m
    hydraulic_diameter: float = 0.0  # m
    cross_section: float = 0.0  # m^2
    roughness: float = 0.0  # m
    friction_factor_model: Optional[FrictionFactorModelType] = None
    friction_multiplier: float = 1.0

    # VALVE: dp = linear_resistance * mdot + quadratic_resistance * mdot*|mdot|.
    linear_resistance: float = 0.0  # Pa/(kg/s)
    quadratic_resistance: float = 0.0  # Pa/(kg/s)^2
    # Optional check-valve behaviour: when the previous-step flow is
    # negative, the branch presents this (large) linear resistance
    # instead of its forward law. None = symmetric valve.
    reverse_linear_resistance: Optional[float] = None  # Pa/(kg/s)

    # RELIEF_VALVE trigger: opens at trip_pressure, reseats below
    # reseat_pressure, both read at monitored_node (None = from_node).
    trip_pressure: Optional[float] = None  # Pa
    reseat_pressure: Optional[float] = None  # Pa
    monitored_node: Optional[str] = None

    # PUMP
    characteristic: Optional[PumpCharacteristic] = None

    # Common. Inertance in the mass-flow formulation is length/area (the
    # density cancels between the momentum storage and the mass flow rate);
    # None defaults to length/cross_section for pipes and 0 (algebraic
    # branch) otherwise.
    inertance: Optional[float] = None  # 1/m
    initial_mass_flow: float = 0.0  # kg/s, positive from from_node to to_node

    def resolved_inertance(self) -> float:
        if self.inertance is not None:
            return self.inertance
        if self.kind is NetworkBranchKind.PIPE:
            return self.length / self.cross_section
        return 0.0


@dataclass
class PortInput:
    """Coupling of a network node to one end of a conductor channel.

    Only the node reference is validated at this level; conductor and
    channel identifiers are resolved when the coupled problem is built
    (stage 1)."""
    node: str
    conductor: str
    channel: str
    end: PortEnd


_TOP_LEVEL_KEYS = {"fluid_type", "nodes", "branches", "ports"}
_RESERVOIR_KEYS = {"identifier", "kind", "pressure", "temperature"}
_INTERNAL_NODE_KEYS = {
    "identifier", "kind", "volume", "initial_pressure", "initial_temperature"
}
_BRANCH_COMMON_KEYS = {
    "identifier", "kind", "from", "to", "inertance", "initial_mass_flow"
}
_PIPE_KEYS = _BRANCH_COMMON_KEYS | {
    "length", "hydraulic_diameter", "cross_section", "roughness",
    "friction_factor_model", "friction_multiplier",
}
_VALVE_KEYS = _BRANCH_COMMON_KEYS | {
    "linear_resistance", "quadratic_resistance", "reverse_linear_resistance"
}
_RELIEF_VALVE_KEYS = _BRANCH_COMMON_KEYS | {
    "linear_resistance", "quadratic_resistance",
    "trip_pressure", "reseat_pressure", "monitored_node",
}
_PUMP_KEYS = _BRANCH_COMMON_KEYS | {"characteristic"}
_PUMP_CHARACTERISTIC_KEYS = {
    "head_at_zero_flow", "linear_coefficient", "quadratic_coefficient"
}
_PORT_KEYS = {"node", "conductor", "channel", "end"}


def _reject_unknown_keys(mapping: dict, allowed_keys: set, context: str):
    unknown = set(mapping) - allowed_keys
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {context}; "
            f"allowed keys are {sorted(allowed_keys)}."
        )


def _require_keys(mapping: dict, required_keys: set, context: str):
    missing = required_keys - set(mapping)
    if missing:
        raise ValueError(
            f"Missing required key(s) {sorted(missing)} in {context}."
        )


def _parse_node(mapping: dict) -> NetworkNodeInput:
    _require_keys(mapping, {"identifier", "kind"}, "hydraulic network node")
    identifier = str(mapping["identifier"])
    kind = get_network_node_kind(mapping["kind"])
    context = f"network node {identifier!r}"
    if kind is NetworkNodeKind.RESERVOIR:
        _reject_unknown_keys(mapping, _RESERVOIR_KEYS, context)
        _require_keys(mapping, {"pressure", "temperature"}, context)
        return NetworkNodeInput(
            identifier=identifier,
            kind=kind,
            pressure=float(mapping["pressure"]),
            temperature=float(mapping["temperature"]),
        )
    _reject_unknown_keys(mapping, _INTERNAL_NODE_KEYS, context)
    _require_keys(mapping, {"initial_pressure", "initial_temperature"}, context)
    return NetworkNodeInput(
        identifier=identifier,
        kind=kind,
        pressure=float(mapping["initial_pressure"]),
        temperature=float(mapping["initial_temperature"]),
        volume=float(mapping.get("volume", 0.0)),
    )


def _parse_branch(mapping: dict) -> NetworkBranchInput:
    _require_keys(
        mapping, {"identifier", "kind", "from", "to"}, "hydraulic network branch"
    )
    identifier = str(mapping["identifier"])
    kind = get_network_branch_kind(mapping["kind"])
    context = f"network branch {identifier!r}"
    common = dict(
        identifier=identifier,
        kind=kind,
        from_node=str(mapping["from"]),
        to_node=str(mapping["to"]),
        inertance=(
            float(mapping["inertance"]) if "inertance" in mapping else None
        ),
        initial_mass_flow=float(mapping.get("initial_mass_flow", 0.0)),
    )
    if kind is NetworkBranchKind.PIPE:
        _reject_unknown_keys(mapping, _PIPE_KEYS, context)
        _require_keys(
            mapping,
            {"length", "hydraulic_diameter", "cross_section",
             "friction_factor_model"},
            context,
        )
        return NetworkBranchInput(
            **common,
            length=float(mapping["length"]),
            hydraulic_diameter=float(mapping["hydraulic_diameter"]),
            cross_section=float(mapping["cross_section"]),
            roughness=float(mapping.get("roughness", 0.0)),
            friction_factor_model=(
                FrictionFactorModelType.get_friction_factor_model(
                    int(mapping["friction_factor_model"])
                )
            ),
            friction_multiplier=float(mapping.get("friction_multiplier", 1.0)),
        )
    if kind is NetworkBranchKind.VALVE:
        _reject_unknown_keys(mapping, _VALVE_KEYS, context)
        reverse = mapping.get("reverse_linear_resistance")
        return NetworkBranchInput(
            **common,
            linear_resistance=float(mapping.get("linear_resistance", 0.0)),
            quadratic_resistance=float(mapping.get("quadratic_resistance", 0.0)),
            reverse_linear_resistance=(
                float(reverse) if reverse is not None else None
            ),
        )
    if kind is NetworkBranchKind.RELIEF_VALVE:
        _reject_unknown_keys(mapping, _RELIEF_VALVE_KEYS, context)
        _require_keys(mapping, {"trip_pressure", "reseat_pressure"}, context)
        monitored = mapping.get("monitored_node")
        return NetworkBranchInput(
            **common,
            linear_resistance=float(mapping.get("linear_resistance", 0.0)),
            quadratic_resistance=float(mapping.get("quadratic_resistance", 0.0)),
            trip_pressure=float(mapping["trip_pressure"]),
            reseat_pressure=float(mapping["reseat_pressure"]),
            monitored_node=str(monitored) if monitored is not None else None,
        )
    # PUMP
    _reject_unknown_keys(mapping, _PUMP_KEYS, context)
    _require_keys(mapping, {"characteristic"}, context)
    characteristic = mapping["characteristic"]
    _reject_unknown_keys(
        characteristic, _PUMP_CHARACTERISTIC_KEYS, f"{context} characteristic"
    )
    _require_keys(
        characteristic, {"head_at_zero_flow"}, f"{context} characteristic"
    )
    return NetworkBranchInput(
        **common,
        characteristic=PumpCharacteristic(
            head_at_zero_flow=float(characteristic["head_at_zero_flow"]),
            linear_coefficient=float(
                characteristic.get("linear_coefficient", 0.0)
            ),
            quadratic_coefficient=float(
                characteristic.get("quadratic_coefficient", 0.0)
            ),
        ),
    )


def _parse_port(mapping: dict) -> PortInput:
    _require_keys(mapping, _PORT_KEYS, "hydraulic network port")
    _reject_unknown_keys(mapping, _PORT_KEYS, "hydraulic network port")
    return PortInput(
        node=str(mapping["node"]),
        conductor=str(mapping["conductor"]),
        channel=str(mapping["channel"]),
        end=get_port_end(mapping["end"]),
    )


@dataclass
class HydraulicNetworkInput:
    """Parsed and validated ``hydraulic_network:`` section."""
    fluid_type: FluidType
    nodes: List[NetworkNodeInput]
    branches: List[NetworkBranchInput]
    ports: List[PortInput] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, mapping: dict) -> "HydraulicNetworkInput":
        """Build the input from the loaded YAML mapping of the
        ``hydraulic_network:`` section and validate it."""
        _reject_unknown_keys(mapping, _TOP_LEVEL_KEYS, "hydraulic_network")
        _require_keys(
            mapping, {"fluid_type", "nodes", "branches"}, "hydraulic_network"
        )
        instance = cls(
            fluid_type=get_fluid_type(mapping["fluid_type"]),
            nodes=[_parse_node(node) for node in mapping["nodes"]],
            branches=[_parse_branch(branch) for branch in mapping["branches"]],
            ports=[_parse_port(port) for port in mapping.get("ports", [])],
        )
        instance.validate()
        return instance

    def validate(self):
        """Check topology and values; raises ValueError with a specific
        message on the first violation found."""
        self._validate_identifiers()
        node_by_identifier = {node.identifier: node for node in self.nodes}
        self._validate_nodes()
        self._validate_branches(node_by_identifier)
        self._validate_connectivity(node_by_identifier)
        self._validate_ports(node_by_identifier)

    def _validate_identifiers(self):
        for kind, identifiers in (
            ("node", [node.identifier for node in self.nodes]),
            ("branch", [branch.identifier for branch in self.branches]),
        ):
            duplicates = {
                identifier for identifier in identifiers
                if identifiers.count(identifier) > 1
            }
            if duplicates:
                raise ValueError(
                    f"Duplicate network {kind} identifier(s): "
                    f"{sorted(duplicates)}."
                )

    def _validate_nodes(self):
        for node in self.nodes:
            context = f"network node {node.identifier!r}"
            if node.pressure <= 0.0:
                raise ValueError(f"{context}: pressure must be positive.")
            if node.temperature <= 0.0:
                raise ValueError(f"{context}: temperature must be positive.")
            if node.volume < 0.0:
                raise ValueError(f"{context}: volume must be non-negative.")

    def _validate_branches(self, node_by_identifier: dict):
        for branch in self.branches:
            context = f"network branch {branch.identifier!r}"
            for endpoint in (branch.from_node, branch.to_node):
                if endpoint not in node_by_identifier:
                    raise ValueError(
                        f"{context}: references unknown node {endpoint!r}."
                    )
            if branch.from_node == branch.to_node:
                raise ValueError(
                    f"{context}: from and to must be different nodes."
                )
            if branch.inertance is not None and branch.inertance < 0.0:
                raise ValueError(f"{context}: inertance must be non-negative.")
            if branch.kind is NetworkBranchKind.PIPE:
                for name in ("length", "hydraulic_diameter", "cross_section"):
                    if getattr(branch, name) <= 0.0:
                        raise ValueError(f"{context}: {name} must be positive.")
                if branch.roughness < 0.0:
                    raise ValueError(
                        f"{context}: roughness must be non-negative."
                    )
                if branch.friction_multiplier <= 0.0:
                    raise ValueError(
                        f"{context}: friction_multiplier must be positive."
                    )
            elif branch.kind is NetworkBranchKind.VALVE:
                if (
                    branch.linear_resistance < 0.0
                    or branch.quadratic_resistance < 0.0
                ):
                    raise ValueError(
                        f"{context}: valve resistances must be non-negative."
                    )
                if branch.linear_resistance + branch.quadratic_resistance == 0.0:
                    raise ValueError(
                        f"{context}: valve needs a positive linear_resistance "
                        "or quadratic_resistance."
                    )
            elif branch.kind is NetworkBranchKind.RELIEF_VALVE:
                if (
                    branch.linear_resistance < 0.0
                    or branch.quadratic_resistance < 0.0
                ):
                    raise ValueError(
                        f"{context}: relief-valve resistances must be "
                        "non-negative."
                    )
                if branch.linear_resistance + branch.quadratic_resistance == 0.0:
                    raise ValueError(
                        f"{context}: relief valve needs a positive open-state "
                        "linear_resistance or quadratic_resistance."
                    )
                if branch.reseat_pressure <= 0.0:
                    raise ValueError(
                        f"{context}: reseat_pressure must be positive."
                    )
                if branch.trip_pressure <= branch.reseat_pressure:
                    raise ValueError(
                        f"{context}: trip_pressure must exceed reseat_pressure "
                        "(the hysteresis band)."
                    )
                if (
                    branch.monitored_node is not None
                    and branch.monitored_node not in node_by_identifier
                ):
                    raise ValueError(
                        f"{context}: monitored_node "
                        f"{branch.monitored_node!r} is not a network node."
                    )
            elif branch.kind is NetworkBranchKind.PUMP:
                characteristic = branch.characteristic
                if characteristic is None:
                    raise ValueError(f"{context}: pump needs a characteristic.")
                if (
                    characteristic.head_at_zero_flow < 0.0
                    or characteristic.linear_coefficient < 0.0
                    or characteristic.quadratic_coefficient < 0.0
                ):
                    raise ValueError(
                        f"{context}: pump characteristic coefficients must be "
                        "non-negative."
                    )

    def _validate_connectivity(self, node_by_identifier: dict):
        """Every connected part of the network must contain a reservoir,
        otherwise its pressure level is undetermined (the MNA ground rule)."""
        adjacency = {identifier: set() for identifier in node_by_identifier}
        for branch in self.branches:
            adjacency[branch.from_node].add(branch.to_node)
            adjacency[branch.to_node].add(branch.from_node)
        visited = set()
        for start in node_by_identifier:
            if start in visited:
                continue
            component = {start}
            frontier = [start]
            while frontier:
                for neighbour in adjacency[frontier.pop()]:
                    if neighbour not in component:
                        component.add(neighbour)
                        frontier.append(neighbour)
            visited |= component
            if not any(
                node_by_identifier[identifier].kind
                is NetworkNodeKind.RESERVOIR
                for identifier in component
            ):
                raise ValueError(
                    "Network part containing node "
                    f"{start!r} has no reservoir: its pressure level would "
                    "be undetermined (add a reservoir node)."
                )

    def _validate_ports(self, node_by_identifier: dict):
        seen = set()
        for port in self.ports:
            if port.node not in node_by_identifier:
                raise ValueError(
                    f"Network port of {port.conductor}/{port.channel} "
                    f"references unknown node {port.node!r}."
                )
            if (
                node_by_identifier[port.node].kind
                is not NetworkNodeKind.INTERNAL
            ):
                raise ValueError(
                    f"Network port of {port.conductor}/{port.channel} must "
                    f"reference an internal node; {port.node!r} is a "
                    "reservoir (a fixed-pressure port is just the ordinary "
                    "imposed-pressure boundary condition)."
                )
            key = (port.conductor, port.channel, port.end)
            if key in seen:
                raise ValueError(
                    f"Duplicate network port for {port.conductor}/"
                    f"{port.channel} end {port.end.value!r}."
                )
            seen.add(key)
