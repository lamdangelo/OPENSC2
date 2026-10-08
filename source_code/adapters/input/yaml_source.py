"""YAML implementation of the input source port (thin wrapper around the
schema-v2 registry, which already serves legacy-shaped dictionaries)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from adapters.input.yaml_input_registry import YamlInputRegistry


class YamlInputSource:
    def __init__(self, registry: YamlInputRegistry):
        self.registry = registry
        self.directory = registry.directory

    # simulation level
    def transient_settings(self) -> Dict[str, Any]:
        return self.registry.transient_settings()

    def environment_settings(self) -> Dict[str, Any]:
        return self.registry.environment_settings()

    def hydraulic_network(self) -> Optional[Dict[str, Any]]:
        return self.registry.hydraulic_network()

    @property
    def conductor_count(self) -> int:
        return self.registry.conductor_count

    def conductor_coupling_dataframe(self):
        return self.registry.conductor_coupling_dataframe()

    # conductor level
    def conductor_name(self, counter: int) -> str:
        return self.registry.conductor_name(counter)

    def conductor_identifier(self, counter: int) -> str:
        return self.registry.conductor_identifier(counter)

    def conductor_files(self, counter: int) -> Dict[str, Any]:
        return self.registry.conductor_files(counter)

    def conductor_inputs(self, counter: int) -> Dict[str, Any]:
        return self.registry.conductor_inputs(counter)

    def conductor_operations(self, counter: int) -> Dict[str, Any]:
        return self.registry.conductor_operations(counter)

    def grid_settings(self, counter: int) -> Dict[str, Any]:
        return self.registry.grid_settings(counter)

    def diagnostic_spatial_times(self, counter: int) -> List[float]:
        return self.registry.diagnostic_spatial_times(counter)

    def diagnostic_time_positions(self, counter: int) -> List[float]:
        return self.registry.diagnostic_time_positions(counter)

    def coupling_dataframes(self, counter: int) -> Dict[str, Any]:
        return self.registry.coupling_dataframes(counter)

    # components (identifiers are unique across the deck; sheet is unused)
    def component_groups(self, counter: int) -> List[Tuple[str, str, List[str]]]:
        return self.registry.component_groups(counter)

    def component_raw_inputs(self, counter, sheet_name, identifier) -> Dict[str, Any]:
        return self.registry.component_raw_inputs(identifier)

    def component_raw_operations(self, counter, sheet_name, identifier) -> Dict[str, Any]:
        return self.registry.component_raw_operations(identifier)
