"""
Convert an OPENSC2 Excel input directory to the YAML input format (schema v1).

Read-only on the workbooks (no openpyxl save, hence no formula-cache damage
and no soffice repair round-trip). Writes ``simulation.yaml`` plus one
``conductor_<IDENTIFIER>.yaml`` per conductor into the same directory (or
``--output-directory``). External time-series files (external_current, ...)
are configuration *data* and stay untouched; the YAML files reference the
directory contents exactly like the workbooks did.

Usage:
    python tools/convert_inputs_to_yaml.py <input_directory> [--output-directory DIR]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source_code"))

import interfaces.yaml_schema_v2 as schema_v2  # noqa: E402
from interfaces.yaml_input_registry import (  # noqa: E402
    ADAPTIVITY_MODE_FROM_VALUE,
    COUPLING_PROPERTY_NAMES,
    SIMULATION_FILE_NAME,
    TIME_STEPPING_KEY_ALIASES,
)


def sanitize(value):
    """Convert pandas/numpy scalars to plain Python types; NaN -> None.

    Also repairs number-as-text workbook cells (e.g. a cross section entered
    as the string '6.2004E-5'): the Excel readers tolerate them because the
    downstream float()/int() casts parse strings, but the YAML files should
    carry real numbers. Genuine text values (material names, flags like
    'BE') are left untouched because float() fails on them.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return value
        if not math.isfinite(number):
            # 'nan'/'inf' strings are not numbers the workbooks mean.
            return value
        if number == int(number) and "." not in text and "e" not in text.lower():
            return int(number)
        return number
    # Empty workbook cells arrive as NaN and must SURVIVE the conversion
    # with their key (the Excel readers deliver them exactly like that, and
    # some inputs are legitimately empty, e.g. an unused ELECTRIC_TIME_STEP);
    # PyYAML round-trips them as '.nan'.
    return value


def rename_keys(mapping, alias_table, drop=()):
    """Emit schema-v2 keys: legacy -> descriptive; unmapped keys unchanged."""
    to_v2 = schema_v2.invert(alias_table)
    return {
        to_v2.get(key, key): value
        for key, value in mapping.items()
        if key not in drop
    }


def split_fluid_boundary_condition(operations):
    """Replace the multiplexed legacy INTIAL of fluid components with the
    v2 pair hydraulic_boundary_condition / boundary_values_from_file."""
    if "INTIAL" not in operations:
        return operations
    operations = dict(operations)
    legacy = int(operations.pop("INTIAL"))
    result = {"hydraulic_boundary_condition": abs(legacy),
              "boundary_values_from_file": legacy < 0}
    result.update(operations)
    return result


def sheet_to_dict(path, sheet_name, value_column, skiprows=2):
    """Read one 'Variable name'/value-column sheet into a dict.

    Reads the cells directly with openpyxl instead of pandas: pandas coerces
    mixed boolean/integer columns inconsistently depending on the dtype
    argument (a TRUE cell can become 1, or an integer 1 become True), while
    openpyxl returns the exact cell values. Empty value cells are kept with
    value None (YAML ``null``); the registry serves them back as NaN, which
    is what the pandas-based Excel readers deliver for empty cells.
    """
    workbook = load_workbook(path, data_only=True, read_only=True)
    rows = list(workbook[sheet_name].iter_rows(values_only=True))
    workbook.close()
    header = rows[skiprows]
    column = header.index(value_column)
    result = {}
    for row in rows[skiprows + 1:]:
        key = row[0] if row else None
        if key is None:
            # Spacer rows have no variable name.
            continue
        value = row[column] if column < len(row) else None
        result[str(key)] = sanitize(value)
    return result


def convert_transient(directory):
    transitory_path = next(directory.glob("*transitory_input*.xlsx"))
    raw = sheet_to_dict(transitory_path, "TRANSIENT", "Value", skiprows=1)

    time_stepping = {}
    for yaml_key, legacy_key in TIME_STEPPING_KEY_ALIASES.items():
        if legacy_key in raw:
            value = raw.pop(legacy_key)
            if yaml_key == "adaptivity":
                value = ADAPTIVITY_MODE_FROM_VALUE.get(int(value), int(value))
            time_stepping[yaml_key] = value

    simulation = {
        "name": raw.pop("SIMULATION"),
        "end_time": float(raw.pop("TEND")),
        "time_stepping": time_stepping,
    }
    magnet_file = raw.pop("MAGNET")
    environment_file = raw.pop("ENVIRONMENT")
    # Anything else in the TRANSIENT sheet (e.g. TIME_SHIFT of the CS3U2
    # schedule) passes through verbatim.
    if raw:
        simulation["legacy_settings"] = raw
    return simulation, magnet_file, environment_file


def convert_environment(directory, environment_file):
    return rename_keys(
        sheet_to_dict(
            directory / environment_file, "ENVIRONMENT", "Value", skiprows=0
        ),
        schema_v2.ENVIRONMENT_KEYS,
    )


def convert_components(directory, structure_path, operation_path):
    workbook = load_workbook(directory / structure_path, data_only=True)
    components = []
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        kind = sheet.cell(row=1, column=1).value
        count = int(sheet.cell(row=1, column=2).value or 0)
        for instance in range(1, count + 1):
            identifier = sheet.cell(row=3, column=4 + instance).value
            operations = sheet_to_dict(
                directory / operation_path, sheet_name, identifier
            )
            if kind == "CHAN":
                operations = split_fluid_boundary_condition(operations)
            components.append(
                {
                    "identifier": identifier,
                    "kind": kind,
                    "sheet": sheet_name,
                    "inputs": rename_keys(
                        sheet_to_dict(
                            directory / structure_path, sheet_name, identifier
                        ),
                        schema_v2.COMPONENT_INPUT_KEYS,
                    ),
                    "operations": rename_keys(
                        operations, schema_v2.COMPONENT_OPERATION_KEYS
                    ),
                }
            )
    return components


def convert_couplings(directory):
    coupling_path = next(
        p for p in directory.glob("*coupling*.xlsx")
        if "external" not in p.name
    )
    sheets = pd.read_excel(
        coupling_path, sheet_name=None, skiprows=1, header=0, index_col=0
    )
    order = sheets[next(iter(sheets))].columns.tolist()
    pair_records = {}
    for name, frame in sheets.items():
        if name not in COUPLING_PROPERTY_NAMES:
            raise ValueError(
                f"{coupling_path.name}: unknown coupling sheet {name!r}."
            )
        matrix = np.triu(frame.to_numpy(dtype=float))
        for i, j in zip(*np.nonzero(matrix)):
            pair = (order[i], order[j])
            record = pair_records.setdefault(pair, {})
            value = matrix[i, j]
            record[name] = int(value) if value == int(value) else float(value)
    couplings = [
        {"between": list(pair),
         **rename_keys(properties, schema_v2.COUPLING_PROPERTY_KEYS)}
        for pair, properties in pair_records.items()
    ]
    return [str(name) for name in order], couplings


def convert_conductor(directory, magnet_file, counter):
    definition_path = directory / magnet_file
    workbook = load_workbook(definition_path, data_only=True)
    files_sheet = workbook[workbook.sheetnames[0]]
    name = files_sheet.cell(row=1, column=1).value
    identifier = files_sheet.cell(row=3, column=4 + counter).value

    conductor_files = sheet_to_dict(
        definition_path, workbook.sheetnames[0], identifier
    )
    document = {
        "conductor": {
            "name": name,
            "identifier": identifier,
            "inputs": rename_keys(
                sheet_to_dict(
                    definition_path, workbook.sheetnames[1], identifier
                ),
                schema_v2.CONDUCTOR_INPUT_KEYS,
                # NAME is redundant with conductor.name one level up.
                drop=("NAME",),
            ),
            "operations": rename_keys(
                sheet_to_dict(
                    definition_path, workbook.sheetnames[2], identifier
                ),
                schema_v2.CONDUCTOR_OPERATION_KEYS,
            ),
        },
        "components": convert_components(
            directory,
            conductor_files["STRUCTURE_ELEMENTS"],
            conductor_files["OPERATION"],
        ),
    }

    order, couplings = convert_couplings(directory)
    document["coupling_component_order"] = order
    document["couplings"] = couplings

    grid_path = next(
        p for p in directory.glob("*grid*.xlsx") if "external" not in p.name
    )
    document["grid"] = rename_keys(
        sheet_to_dict(grid_path, "GRID", identifier), schema_v2.GRID_KEYS
    )

    diagnostic_path = next(directory.glob("*diagnostic*.xlsx"))
    document["diagnostics"] = {
        "spatial_distribution_times": [
            float(v)
            for v in pd.read_excel(
                diagnostic_path,
                sheet_name="Spatial_distribution",
                skiprows=2,
                header=0,
                usecols=[identifier],
            )[identifier].dropna()
        ],
        "time_evolution_positions": [
            float(v)
            for v in pd.read_excel(
                diagnostic_path,
                sheet_name="Time_evolution",
                skiprows=2,
                header=0,
                usecols=[identifier],
            )[identifier].dropna()
        ],
    }
    return identifier, document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()
    directory = args.input_directory
    output = args.output_directory or directory
    output.mkdir(parents=True, exist_ok=True)

    simulation, magnet_file, environment_file = convert_transient(directory)
    definition = load_workbook(directory / magnet_file, data_only=True)
    conductor_count = int(
        definition[definition.sheetnames[0]].cell(row=1, column=2).value
    )

    conductors = []
    for counter in range(1, conductor_count + 1):
        identifier, document = convert_conductor(directory, magnet_file, counter)
        file_name = f"conductor_{identifier}.yaml"
        with open(output / file_name, "w") as stream:
            stream.write(
                "# OPENSC2 conductor input (YAML schema v2), converted from "
                f"{magnet_file}.\n"
            )
            yaml.safe_dump(
                document, stream, sort_keys=False, default_flow_style=False
            )
        conductors.append({"file": file_name})
        print(f"wrote {output / file_name}")

    top = {
        "format": "opensc2-yaml/1",
        "simulation": simulation,
        "environment": convert_environment(directory, environment_file),
        "conductors": conductors,
    }
    # Conductor-to-conductor coupling (only meaningful for multi-conductor
    # magnets; skip when everything is zero).
    coupling_sheet = pd.read_excel(
        directory / magnet_file,
        sheet_name="CONDUCTOR_coupling",
        header=0,
        index_col=0,
    )
    matrix = np.triu(coupling_sheet.to_numpy(dtype=float))
    records = [
        {
            "between": [
                str(coupling_sheet.columns[i]),
                str(coupling_sheet.columns[j]),
            ],
            "value": float(matrix[i, j]),
        }
        for i, j in zip(*np.nonzero(matrix))
    ]
    if records:
        top["conductor_couplings"] = records

    with open(output / SIMULATION_FILE_NAME, "w") as stream:
        stream.write(
            "# OPENSC2 simulation input (YAML schema v2), converted from the "
            "Excel workbooks.\n"
        )
        yaml.safe_dump(top, stream, sort_keys=False, default_flow_style=False)
    print(f"wrote {output / SIMULATION_FILE_NAME}")


if __name__ == "__main__":
    main()
