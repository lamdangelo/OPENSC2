"""Input adapters: readers of the YAML and (frozen) Excel decks and the
builders that turn raw sheet dictionaries into the typed input dataclasses."""

from __future__ import annotations

from pathlib import Path

from adapters.input import yaml_input_registry
from adapters.input.excel_source import ExcelInputSource
from adapters.input.yaml_source import YamlInputSource


def open_input_source(directory):
    """The input source for a deck directory: YAML when it holds
    simulation.yaml, otherwise the legacy Excel workbooks."""
    registry = yaml_input_registry.get_registry(Path(directory))
    if registry is not None:
        return YamlInputSource(registry)
    return ExcelInputSource(directory)
