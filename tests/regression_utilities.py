"""Shared helpers for the TDD-example regression tests.

Each regression test copies the input files of one TDD example into a
temporary directory, runs the full simulation headless there, and compares
the produced outputs against committed reference data within a column-scaled
tolerance (robust against floating-point noise from different BLAS/CPU
combinations on CI runners, while still catching real physics or solver
regressions, which show up orders of magnitude above the tolerance).

To regenerate reference data after an intentional physics/solver change, run
the simulation on a copy of the example's input directory and copy the
corresponding files from ``<method>/<simulation name>/Output``.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Maximum allowed deviation, relative to the column-wise maximum magnitude of
# the reference data. Cross-platform floating-point noise stays orders of
# magnitude below this; genuine regressions observed so far sit at 1e-4 and
# above.
RELATIVE_TOLERANCE = 1.0e-5

INPUT_FILE_PATTERNS = ("*.xlsx", "*.yaml")


def copy_input_files(
    source_directory: Path, target_directory: Path, input_format: str
) -> None:
    """Copy the workbook and YAML input files of a TDD example.

    For ``input_format == "excel"`` the YAML files are omitted, forcing the
    legacy Excel input path (a directory containing simulation.yaml is
    always YAML-driven).
    """
    for pattern in INPUT_FILE_PATTERNS:
        for input_file in source_directory.glob(pattern):
            shutil.copy2(input_file, target_directory / input_file.name)
    if input_format == "excel":
        for yaml_file in target_directory.glob("*.yaml"):
            yaml_file.unlink()


def discover_output_directory(run_directory: Path) -> Path:
    """Return the Output directory of the single simulation run inside
    run_directory (tree layout: <method>/<simulation name>/Output)."""
    solution_directories = list(
        run_directory.glob("*/*/Output/Solution/CONDUCTOR_1")
    )
    assert len(solution_directories) == 1, (
        "expected exactly one simulation output tree, found: "
        f"{solution_directories}"
    )
    return solution_directories[0].parents[1]


def compare_against_reference(output_file: Path, reference_file: Path) -> None:
    """Assert that output_file matches reference_file within tolerance."""
    reference = pd.read_csv(reference_file, sep="\t")
    output = pd.read_csv(output_file, sep="\t")
    assert list(output.columns) == list(reference.columns), (
        f"{output_file.name}: column names changed with respect to the "
        f"reference ({list(output.columns)} vs {list(reference.columns)})"
    )
    assert output.shape == reference.shape, (
        f"{output_file.name}: shape {output.shape} differs from reference "
        f"shape {reference.shape}"
    )
    reference_values = reference.to_numpy(dtype=float)
    output_values = output.to_numpy(dtype=float)
    assert not np.isnan(output_values).any(), f"{output_file.name}: NaN in output"
    column_scale = np.maximum(
        np.abs(reference_values).max(axis=0), np.finfo(float).tiny
    )
    relative_deviation = np.abs(output_values - reference_values) / column_scale
    worst_column = np.unravel_index(
        np.argmax(relative_deviation), relative_deviation.shape
    )[1]
    assert relative_deviation.max() <= RELATIVE_TOLERANCE, (
        f"{output_file.name}: maximum column-scaled deviation "
        f"{relative_deviation.max():.3e} exceeds tolerance "
        f"{RELATIVE_TOLERANCE:.1e} (worst column: "
        f"'{reference.columns[worst_column]}')"
    )


def compare_output_tree(output_directory: Path, reference_directory: Path) -> None:
    """Compare every committed reference file (Solution and Time_evolution)
    against its counterpart in the run's Output directory."""
    for section in ("Solution", "Time_evolution"):
        for reference_file in sorted((reference_directory / section).iterdir()):
            compare_against_reference(
                output_directory / section / "CONDUCTOR_1" / reference_file.name,
                reference_file,
            )
