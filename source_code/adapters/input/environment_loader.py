"""Reads the environment inputs (YAML ``environment:`` section or the legacy
ENVIRONMENT workbook sheet) and builds the domain Environment with a
CoolProp-backed property provider for its medium."""

from __future__ import annotations

import pandas as pd

import adapters.input.yaml_input_registry as yaml_input_registry
from adapters.fluid_properties.coolprop_provider import CoolPropNamedFluidProperties
from environment.environment import Environment, EnvironmentSettings


def load_environment_settings(file_path) -> EnvironmentSettings:
    registry = yaml_input_registry.get_registry(file_path)
    if registry is not None:
        raw = registry.environment_settings()
    else:
        raw = pd.read_excel(
            file_path,
            sheet_name="ENVIRONMENT",
            header=0,
            index_col=0,
            usecols=["Variable name", "Value"],
        )["Value"].to_dict()
    return EnvironmentSettings(
        medium=str(raw["Medium"]),
        temperature=raw["Temperature"],
        pressure=raw["Pressure"],
    )


def load_environment(file_path) -> Environment:
    settings = load_environment_settings(file_path)
    return Environment(
        settings, CoolPropNamedFluidProperties(settings.medium.lower())
    )
