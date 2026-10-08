"""File-backed implementation of ports.external_inputs.ExternalInputReader."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from adapters.input.auxiliary_files import (
    get_from_xlsx,
    load_auxiliary_files,
    with_read_csv,
    with_read_excel,
)
from conductor.conductor_inputs import ConductorInputFiles
from numerics.interpolation import build_interpolator
from ports.external_inputs import TabulatedSeries

# Reader and default delimiter per file extension (formerly
# Simulation.func_open_aux / Simulation.default_vals).
_READERS = dict(dat=with_read_csv, csv=with_read_csv, tsv=with_read_csv, xlsx=with_read_excel)
_DEFAULT_DELIMITERS = dict(dat="\t", csv=";", tsv="\t", xlsx=0)

SCALING_INPUT_FILE = "scaling_input.dat"
_SCALING_INTEGER_KEYS = ("emode", "ieavside")


class FileExternalInputReader:
    def __init__(self, file_paths: ConductorInputFiles):
        self.file_paths = file_paths

    def tabulated_series(self, quantity, identifier, interpolation_kind):
        path = getattr(self.file_paths, f"external_{quantity}")
        frame, unit_flag = load_auxiliary_files(path, sheetname=identifier)
        interpolator, interpolation_flag = build_interpolator(frame, interpolation_kind)
        return TabulatedSeries(interpolator, interpolation_flag, unit_flag)

    def scaling_input(self) -> dict:
        # Looked up next to the deck first; the legacy location was the
        # current working directory, which is kept as fallback.
        candidates = [Path(self.file_paths.base_path) / SCALING_INPUT_FILE, Path(os.getcwd()) / SCALING_INPUT_FILE]
        path = next((p for p in candidates if p.exists()), candidates[-1])
        scaling = {}
        with open(path, "r") as stream:
            for line in stream:
                if line[0] == "#" or line[0] == "\n":
                    continue
                fields = line.split(" = ")
                if fields[0] in _SCALING_INTEGER_KEYS:
                    scaling[fields[0]] = int(fields[1][:-1])
                elif fields[0] == "superconducting_material":
                    scaling[fields[0]] = str(fields[1][:-1])
                else:
                    scaling[fields[0]] = float(fields[1][:-1])
        return scaling

    def read_flow_parameters(self, conductor, path, component, flag_name, *intial):
        return get_from_xlsx(conductor, path, component, flag_name, *intial)

    def user_defined_quantity(self, quantity: str, sheet_name: str):
        file_name = getattr(self.file_paths, quantity.lower())
        extension = str(file_name).split(".")[1]
        return _READERS[extension](file_name, sheet_name, _DEFAULT_DELIMITERS[extension])

    def user_defined_grid_tables(self) -> dict:
        return pd.read_excel(self.file_paths.external_grid, sheet_name=None)
