"""
YAML input source for OPENSC2 (schema v1).

A simulation directory is YAML-driven when it contains ``simulation.yaml``.
The registry parses that file (and the per-conductor files it references)
once, and serves every configuration read of the run as the same plain
dictionaries / data frames the Excel readers produce, so the downstream
dataclass constructors and validators are untouched.

Schema v1 keeps the legacy workbook variable names as leaf keys inside the
``inputs:`` / ``operations:`` / ``grid:`` blocks (the converter emits them
verbatim); the *structure* is native YAML: one document per conductor,
components as a list, couplings as a sparse symmetric pair list instead of
eleven triangular matrix sheets, diagnostics as plain lists. Renaming the
leaf keys to descriptive names via an alias table is the planned schema v2.

External time-series files (current waveform, external flow, ...) are not
part of this migration stage: they are data, not configuration, and both
CSV and XLSX readers for them already exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import interfaces.yaml_schema_v2 as schema_v2

SIMULATION_FILE_NAME = "simulation.yaml"

# Names accepted for the time-stepping adaptivity mode (legacy IADAPTIME).
ADAPTIVITY_MODE_NAMES = {
    "FIXED": 0,
    "LEGACY_EIGENVALUE": 1,
    "LEGACY_EIGENVALUE_SOLID_ONLY": 2,
    "CS3U2_SCHEDULE": 3,
    "ERROR_CONTROLLED": 4,
}
ADAPTIVITY_MODE_FROM_VALUE = {
    value: name for name, value in ADAPTIVITY_MODE_NAMES.items()
}

# Descriptive keys of the simulation.time_stepping block -> legacy TRANSIENT
# sheet variable names.
TIME_STEPPING_KEY_ALIASES = {
    "adaptivity": "IADAPTIME",
    "minimum_step": "STPMIN",
    "maximum_step": "STPMAX",
    "tolerance": "TOLTIME",
    "reference_time": "TIMEREF",
    "reference_duration": "TAUREF",
}

# Sheet names of the coupling workbook; a coupling pair record may carry any
# of these as keys (missing means zero, i.e. uncoupled for that property).
COUPLING_PROPERTY_NAMES = (
    "contact_perimeter_flag",
    "contact_perimeter",
    "HTC_choice",
    "contact_HTC",
    "thermal_contact_resistance",
    "HTC_multiplier",
    "electric_conductance_mode",
    "electric_conductance",
    "open_perimeter_fract",
    "interf_thickness",
    "trans_transp_multiplier",
    "view_factors",
)

def _with_excel_empty_cell_semantics(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Serve YAML ``null`` leaves as NaN: the pandas-based Excel readers
    deliver empty value cells as NaN with the key present, and downstream
    code relies on that (e.g. an unused ELECTRIC_TIME_STEP)."""
    return {
        key: (float("nan") if value is None else value)
        for key, value in mapping.items()
    }


_registry_cache: Dict[str, Optional["YamlInputRegistry"]] = {}


def get_registry(directory: Any) -> Optional["YamlInputRegistry"]:
    """Return the registry for ``directory`` if it is YAML-driven, else None.

    ``directory`` may also be a path to a file inside the directory (the
    component loaders only know their workbook path).
    """
    path = Path(directory)
    if path.is_file() or path.suffix:
        path = path.parent
    key = str(path.resolve())
    if key not in _registry_cache:
        simulation_file = path / SIMULATION_FILE_NAME
        if simulation_file.is_file():
            _registry_cache[key] = YamlInputRegistry(simulation_file)
        else:
            _registry_cache[key] = None
    return _registry_cache[key]


class YamlInputRegistry:
    """Parsed view of a YAML-driven input directory (schema v1)."""

    def __init__(self, simulation_file: Path):
        self.directory = simulation_file.parent
        self.simulation_file = simulation_file
        with open(simulation_file) as stream:
            self.document = yaml.safe_load(stream)
        if not isinstance(self.document, dict) or "simulation" not in self.document:
            raise ValueError(
                f"{simulation_file}: not a valid OPENSC2 simulation.yaml "
                "(missing top-level 'simulation' block)."
            )
        self.conductor_documents: List[Dict[str, Any]] = []
        self.conductor_file_names: List[str] = []
        for entry in self.document.get("conductors", []):
            file_name = entry["file"]
            with open(self.directory / file_name) as stream:
                self.conductor_documents.append(yaml.safe_load(stream))
            self.conductor_file_names.append(file_name)
        if not self.conductor_documents:
            raise ValueError(f"{simulation_file}: no conductors declared.")
        # Component lookup by identifier (identifiers must be unique across
        # the simulation, which the converter guarantees).
        self._component_by_identifier: Dict[str, Dict[str, Any]] = {}
        for conductor_document in self.conductor_documents:
            for component in conductor_document.get("components", []):
                identifier = component["identifier"]
                if identifier in self._component_by_identifier:
                    raise ValueError(
                        f"Duplicate component identifier {identifier!r} in "
                        f"{simulation_file.parent}."
                    )
                self._component_by_identifier[identifier] = component

    # ----------------------------------------------------------------- #
    # Simulation level                                                   #
    # ----------------------------------------------------------------- #

    def transient_settings(self) -> Dict[str, Any]:
        """Legacy TRANSIENT-sheet dictionary."""
        simulation = self.document["simulation"]
        time_stepping = simulation.get("time_stepping", {})
        settings: Dict[str, Any] = {
            "SIMULATION": simulation["name"],
            # In YAML mode these point back at the YAML entry file; the
            # registry serves the actual content, the names only feed logs
            # and path joins that must resolve to an existing file.
            "MAGNET": SIMULATION_FILE_NAME,
            "ENVIRONMENT": SIMULATION_FILE_NAME,
            "TEND": float(simulation["end_time"]),
        }
        for yaml_key, legacy_key in TIME_STEPPING_KEY_ALIASES.items():
            if yaml_key not in time_stepping:
                continue
            value = time_stepping[yaml_key]
            if yaml_key == "adaptivity" and isinstance(value, str):
                value = ADAPTIVITY_MODE_NAMES[value]
            settings[legacy_key] = value
        # Legacy settings without a descriptive alias (e.g. TIME_SHIFT of the
        # CS3U2 schedule) pass through verbatim.
        settings.update(simulation.get("legacy_settings", {}))
        return settings

    def environment_settings(self) -> Dict[str, Any]:
        """Legacy ENVIRONMENT-sheet dictionary (v2 keys translated back)."""
        return schema_v2.translate_to_legacy(
            self.document["environment"], schema_v2.ENVIRONMENT_KEYS
        )

    @property
    def conductor_count(self) -> int:
        return len(self.conductor_documents)

    def conductor_coupling_dataframe(self) -> pd.DataFrame:
        """Conductor-to-conductor coupling matrix (legacy CONDUCTOR_coupling
        sheet); zeros unless declared under ``conductor_couplings:``."""
        names = [
            document["conductor"]["identifier"]
            for document in self.conductor_documents
        ]
        frame = pd.DataFrame(
            np.zeros((len(names), len(names))), index=names, columns=names
        )
        for record in self.document.get("conductor_couplings", []):
            first, second = record["between"]
            frame.loc[first, second] = record["value"]
            frame.loc[second, first] = record["value"]
        return frame

    # ----------------------------------------------------------------- #
    # Conductor level (counter is the legacy 1-based conductor index)     #
    # ----------------------------------------------------------------- #

    def _conductor(self, counter: int) -> Dict[str, Any]:
        return self.conductor_documents[counter - 1]["conductor"]

    def _conductor_document(self, counter: int) -> Dict[str, Any]:
        return self.conductor_documents[counter - 1]

    def conductor_name(self, counter: int) -> str:
        return self._conductor(counter)["name"]

    def conductor_identifier(self, counter: int) -> str:
        return self._conductor(counter)["identifier"]

    def conductor_inputs(self, counter: int) -> Dict[str, Any]:
        return _with_excel_empty_cell_semantics(
            schema_v2.translate_to_legacy(
                self._conductor(counter)["inputs"],
                schema_v2.CONDUCTOR_INPUT_KEYS,
            )
        )

    def conductor_operations(self, counter: int) -> Dict[str, Any]:
        return _with_excel_empty_cell_semantics(
            schema_v2.translate_to_legacy(
                self._conductor(counter)["operations"],
                schema_v2.CONDUCTOR_OPERATION_KEYS,
            )
        )

    def conductor_files(self, counter: int) -> Dict[str, Any]:
        """Legacy CONDUCTOR_files dictionary: the structure-elements and
        operation 'workbooks' both resolve to the conductor YAML file."""
        return {
            "STRUCTURE_ELEMENTS": self.conductor_file_names[counter - 1],
            "OPERATION": self.conductor_file_names[counter - 1],
        }

    def grid_settings(self, counter: int) -> Dict[str, Any]:
        return _with_excel_empty_cell_semantics(
            schema_v2.translate_to_legacy(
                self._conductor_document(counter)["grid"], schema_v2.GRID_KEYS
            )
        )

    def diagnostic_spatial_times(self, counter: int) -> List[float]:
        return list(
            self._conductor_document(counter)["diagnostics"][
                "spatial_distribution_times"
            ]
        )

    def diagnostic_time_positions(self, counter: int) -> List[float]:
        return list(
            self._conductor_document(counter)["diagnostics"][
                "time_evolution_positions"
            ]
        )

    # ----------------------------------------------------------------- #
    # Components                                                          #
    # ----------------------------------------------------------------- #

    def component_groups(
        self, counter: int
    ) -> List[Tuple[str, str, List[str]]]:
        """Component instances grouped by kind, preserving declaration order:
        list of (kind, legacy sheet name, [identifiers])."""
        groups: List[Tuple[str, str, List[str]]] = []
        for component in self._conductor_document(counter)["components"]:
            kind = component["kind"]
            sheet = component["sheet"]
            if groups and groups[-1][0] == kind and groups[-1][1] == sheet:
                groups[-1][2].append(component["identifier"])
            else:
                groups.append((kind, sheet, [component["identifier"]]))
        return groups

    def component_raw_inputs(self, identifier: str) -> Dict[str, Any]:
        return _with_excel_empty_cell_semantics(
            schema_v2.translate_to_legacy(
                self._component_by_identifier[identifier]["inputs"],
                schema_v2.COMPONENT_INPUT_KEYS,
            )
        )

    def component_raw_operations(self, identifier: str) -> Dict[str, Any]:
        return _with_excel_empty_cell_semantics(
            schema_v2.translate_to_legacy(
                schema_v2.recompose_legacy_intial(
                    self._component_by_identifier[identifier]["operations"]
                ),
                schema_v2.COMPONENT_OPERATION_KEYS,
            )
        )

    # ----------------------------------------------------------------- #
    # Couplings                                                           #
    # ----------------------------------------------------------------- #

    def coupling_dataframes(self, counter: int) -> Dict[str, pd.DataFrame]:
        """Rebuild the legacy coupling-workbook sheets (one symmetric-upper
        DataFrame per property) from the sparse pair list."""
        document = self._conductor_document(counter)
        order = document["coupling_component_order"]
        index = {name: position for position, name in enumerate(order)}
        matrices = {
            name: np.zeros((len(order), len(order)))
            for name in COUPLING_PROPERTY_NAMES
        }
        for record in document.get("couplings", []):
            record = schema_v2.translate_to_legacy(
                record, schema_v2.COUPLING_PROPERTY_KEYS
            )
            first, second = record["between"]
            try:
                i, j = index[first], index[second]
            except KeyError as error:
                raise ValueError(
                    f"Coupling pair {record['between']} names a component "
                    f"missing from coupling_component_order {order}."
                ) from error
            if i > j:
                i, j = j, i
            for name, value in record.items():
                if name == "between":
                    continue
                if name not in matrices:
                    raise ValueError(
                        f"Unknown coupling property {name!r} for pair "
                        f"{record['between']}; expected one of "
                        f"{COUPLING_PROPERTY_NAMES}."
                    )
                matrices[name][i, j] = value
        return {
            name: pd.DataFrame(matrix, index=order, columns=order)
            for name, matrix in matrices.items()
        }
