"""Input source port.

Everything the model needs from the input deck, independent of its format.
The methods mirror the legacy workbook layout (sheet-shaped dictionaries
keyed by the legacy variable names) so that the existing dataclass builders
keep working unchanged; ``adapters.input.yaml_source`` and the frozen
``adapters.input.excel_source`` implement it, and
``adapters.input.open_input_source`` picks one for a deck directory.
``counter`` is the legacy 1-based conductor index.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple


class InputSource(Protocol):
    # -- simulation level -------------------------------------------------
    def transient_settings(self) -> Dict[str, Any]: ...

    def environment_settings(self) -> Dict[str, Any]: ...

    def hydraulic_network(self) -> Optional[Dict[str, Any]]:
        """Raw ``hydraulic_network:`` mapping, or None when absent."""
        ...

    @property
    def conductor_count(self) -> int: ...

    def conductor_coupling_dataframe(self) -> Any:
        """Conductor-to-conductor thermal contact matrix (pandas DataFrame
        indexed by conductor identifier)."""
        ...

    # -- conductor level --------------------------------------------------
    def conductor_name(self, counter: int) -> str: ...

    def conductor_identifier(self, counter: int) -> str: ...

    def conductor_files(self, counter: int) -> Dict[str, Any]: ...

    def conductor_inputs(self, counter: int) -> Dict[str, Any]: ...

    def conductor_operations(self, counter: int) -> Dict[str, Any]: ...

    def grid_settings(self, counter: int) -> Dict[str, Any]: ...

    def diagnostic_spatial_times(self, counter: int) -> List[float]: ...

    def diagnostic_time_positions(self, counter: int) -> List[float]: ...

    def coupling_dataframes(self, counter: int) -> Dict[str, Any]:
        """{coupling property name: DataFrame} (legacy coupling workbook)."""
        ...

    # -- components -------------------------------------------------------
    def component_groups(self, counter: int) -> List[Tuple[str, str, List[str]]]:
        """[(kind, legacy sheet name, [identifiers])] in declaration order."""
        ...

    def component_raw_inputs(
        self, counter: int, sheet_name: str, identifier: str
    ) -> Dict[str, Any]: ...

    def component_raw_operations(
        self, counter: int, sheet_name: str, identifier: str
    ) -> Dict[str, Any]: ...
