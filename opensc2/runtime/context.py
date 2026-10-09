"""The running simulation as the domain and the application see it.

``SimulationContext`` carries the shared state of a run (settings, clock,
conductors, environment, network, sources, output paths) without any of the
facade's wiring (input sources, observers, summary). Physics functions that
used to take the ``Simulation`` object take the context. The legacy
attribute names (``transient_input``, ``list_of_Conductors``, ``dict_path``,
``dict_qsource``) are kept as read/write aliases because drivers and the
output adapters still use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opensc2.ports.diagnostics import NULL_DIAGNOSTICS


@dataclass
class SimulationContext:
    transient_settings: Dict[str, Any]  # legacy TRANSIENT dict (drivers write to it)
    environment: Any
    base_path: str = ""
    conductors: List[Any] = field(default_factory=list)
    simulation_time: List[float] = field(default_factory=lambda: [0.0])
    simulation_time_step: float = 0.0
    num_step: int = 0
    n_digit_time: int = 0  # digits used when labelling save times
    hydraulic_network: Optional[Any] = None
    qsources: Dict[str, Any] = field(default_factory=dict)
    conductor_contacts: Any = None  # conductor-to-conductor thermal contact matrix
    output_paths: Dict[str, str] = field(default_factory=dict)
    diagnostics: Any = NULL_DIAGNOSTICS

    # -- legacy aliases ---------------------------------------------------
    @property
    def transient_input(self) -> Dict[str, Any]:
        return self.transient_settings

    @transient_input.setter
    def transient_input(self, value):
        self.transient_settings = value

    @property
    def list_of_Conductors(self) -> List[Any]:
        return self.conductors

    @list_of_Conductors.setter
    def list_of_Conductors(self, value):
        self.conductors = value

    @property
    def dict_qsource(self) -> Dict[str, Any]:
        return self.qsources

    @dict_qsource.setter
    def dict_qsource(self, value):
        self.qsources = value

    @property
    def dict_path(self) -> Dict[str, str]:
        return self.output_paths

    @dict_path.setter
    def dict_path(self, value):
        self.output_paths = value

    @property
    def contactBetweenConductors(self):
        return self.conductor_contacts

    @contactBetweenConductors.setter
    def contactBetweenConductors(self, value):
        self.conductor_contacts = value

    @property
    def basePath(self) -> str:
        return self.base_path

    @property
    def numObj(self) -> int:
        return len(self.conductors)
