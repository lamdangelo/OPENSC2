"""Input plumbing of the ELECTRIC_CURRENT_CONSISTENCY flag.

The steady-state current-consistency Picard iteration
(electromagnetics.electric_solver.solve_steady_state) is opt-in via the
conductor-operations key ``electric_current_consistency`` (legacy alias
``ELECTRIC_CURRENT_CONSISTENCY``), default off so every existing deck is
bit-identical.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source_code"))

import numpy as np  # noqa: E402

from conductor.conductor_inputs import ConductorOperations  # noqa: E402
from conductor.input_loader import ConductorInputLoader  # noqa: E402
import interfaces.yaml_schema_v2 as schema_v2  # noqa: E402


RAW_OPERATIONS = {
    "EQUIPOTENTIAL_SURFACE_FLAG": 0,
    "EQUIPOTENTIAL_SURFACE_NUMBER": 1,
    "EQUIPOTENTIAL_SURFACE_COORDINATE": 0,
    "MAXIMUM_ITERATION_NUMBER": 1000,
    "INDUCTANCE_MODE": 1,
    "SELF_INDUCTANCE_MODE": 2,
    "ELECTRIC_SOLVER": 0,
}


def _build_operations(raw):
    loader = ConductorInputLoader.__new__(ConductorInputLoader)
    return loader.build_conductor_operations(raw)


def test_flag_defaults_to_false_in_dataclass():
    operations = _build_operations(dict(RAW_OPERATIONS))
    assert operations.electric_current_consistency is False


def test_flag_absent_from_raw_inputs_is_false():
    raw = dict(RAW_OPERATIONS)
    assert "ELECTRIC_CURRENT_CONSISTENCY" not in raw
    operations = _build_operations(raw)
    assert operations.electric_current_consistency is False


def test_flag_true_is_read():
    raw = dict(RAW_OPERATIONS, ELECTRIC_CURRENT_CONSISTENCY=True)
    operations = _build_operations(raw)
    assert operations.electric_current_consistency is True


def test_yaml_alias_translates_to_legacy_key():
    translated = schema_v2.translate_to_legacy(
        {"electric_current_consistency": True},
        schema_v2.CONDUCTOR_OPERATION_KEYS,
    )
    assert translated == {"ELECTRIC_CURRENT_CONSISTENCY": True}


def test_equipotential_coordinates_accept_yaml_list():
    operations = _build_operations(
        dict(RAW_OPERATIONS, EQUIPOTENTIAL_SURFACE_COORDINATE=[0.0, 888.0])
    )
    np.testing.assert_allclose(
        operations.equipotential_surface_coordinates, [0.0, 888.0]
    )
