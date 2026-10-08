"""Typed, format-independent definition of one conductor: everything the
Conductor constructor needs, produced by adapters.input.conductor_loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np

from components.component_flags import ComponentType
from conductor.conductor_inputs import (
    ConductorCoupling,
    ConductorInputFiles,
    ConductorInputs,
    ConductorOperations,
)
from conductor.conductor_mesh import ConductorMesh


@dataclass
class ComponentDefinition:
    component_type: ComponentType
    kind: str  # legacy kind string (e.g. "STR_MIX"), kept as component.name
    sheet_name: str
    identifier: str
    inputs: Any
    operations: Any


@dataclass
class ConductorDefinition:
    counter: int  # legacy 1-based index
    name: str
    identifier: str
    inputs: ConductorInputs
    operations: ConductorOperations
    file_paths: ConductorInputFiles
    mesh: ConductorMesh
    coupling: ConductorCoupling
    spatial_save_times: np.ndarray  # diagnostics: save spatial distributions at
    time_save_positions: np.ndarray  # diagnostics: save time evolutions at
    components: List[ComponentDefinition] = field(default_factory=list)
    definition_file: Path = None  # legacy workbook_name
    sheet_names: List[str] = field(default_factory=list)
