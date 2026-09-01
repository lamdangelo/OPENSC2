"""Tests of the YAML input-registry directory resolution.

Regression coverage for the suffix heuristic of get_registry: a model
directory whose name contains a dot (e.g. "..._return1.8L") must not be
mistaken for a file path, while (possibly non-existent) workbook paths
inside a YAML-driven directory still resolve to that directory.
"""

from pathlib import Path

import interfaces.yaml_input_registry as yaml_input_registry

from regression_utilities import copy_input_files

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_1_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "TDD_examples" / "CASE_1_ITER_like_LTS"
)


def make_yaml_model_directory(tmp_path, name: str) -> Path:
    model_directory = tmp_path / name
    model_directory.mkdir()
    copy_input_files(CASE_1_INPUT_DIRECTORY, model_directory, "yaml")
    return model_directory


def test_get_registry_accepts_directory_with_dotted_name(tmp_path):
    model_directory = make_yaml_model_directory(tmp_path, "model_return1.8L")
    assert yaml_input_registry.get_registry(model_directory) is not None
    # The headless drivers pass the directory as a string with a trailing
    # separator.
    assert yaml_input_registry.get_registry(str(model_directory) + "/") is not None


def test_get_registry_resolves_workbook_path_to_parent(tmp_path):
    model_directory = make_yaml_model_directory(tmp_path, "plain_model")
    for workbook_name in ("conductor_definition.xlsx", "missing_workbook.xlsx"):
        registry = yaml_input_registry.get_registry(
            model_directory / workbook_name
        )
        assert registry is not None
        assert registry.directory == model_directory


def test_get_registry_none_for_non_yaml_directory(tmp_path):
    plain_directory = tmp_path / "no_yaml_here.8L"
    plain_directory.mkdir()
    assert yaml_input_registry.get_registry(plain_directory) is None
