"""Regression test for TDD example 1 (CASE_1_ITER_like_LTS).

Runs the full heat-slug transient (TEND = 100 s, 200 elements, backward
Euler, ~1 minute wall time) headless in a temporary directory and compares
the produced outputs against the committed reference data in
``tests/reference_data/CASE_1_ITER_like_LTS``:

* ``Solution``: the complete final solution state of every component
  (channels, mixed strand, jacket) at TEND;
* ``Time_evolution``: temperature, pressure and inlet/outlet histories at
  the user-defined probe positions, covering the whole transient.

The test is parametrized over both supported input formats (YAML schema v2
and the legacy Excel workbooks), which also guards the equivalence of the
two input paths.

The comparison uses a column-scaled tolerance (each column must match within
RELATIVE_TOLERANCE times the column's maximum magnitude) rather than a
bitwise check, so the test is robust against floating-point noise from
different BLAS/CPU combinations on CI runners while still catching real
physics or solver regressions, which show up orders of magnitude above the
tolerance.

The reference data was generated with this same configuration; to regenerate
it after an intentional physics/solver change, run the simulation on a copy
of the input directory and copy the corresponding output files (see the
paths above), e.g.:

    python -c "from simulation import Simulation; \
               Simulation('<copy of TDD_examples/CASE_1_ITER_like_LTS>').run()"
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulation import Simulation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_1_INPUT_DIRECTORY = REPOSITORY_ROOT / "TDD_examples" / "CASE_1_ITER_like_LTS"
REFERENCE_DIRECTORY = Path(__file__).parent / "reference_data" / "CASE_1_ITER_like_LTS"

# Maximum allowed deviation, relative to the column-wise maximum magnitude of
# the reference data. Cross-platform floating-point noise stays orders of
# magnitude below this; genuine regressions observed so far sit at 1e-4 and
# above.
RELATIVE_TOLERANCE = 1.0e-5

INPUT_FILE_PATTERNS = ("*.xlsx", "*.yaml")


@pytest.fixture(params=["yaml", "excel"])
def case_1_run_directory(request, tmp_path):
    """Copy the CASE_1 input files into a temporary directory.

    For the ``excel`` variant the YAML files are removed, forcing the legacy
    Excel input path (a directory containing simulation.yaml is always
    YAML-driven).
    """
    run_directory = tmp_path / "CASE_1_ITER_like_LTS"
    run_directory.mkdir()
    for pattern in INPUT_FILE_PATTERNS:
        for input_file in CASE_1_INPUT_DIRECTORY.glob(pattern):
            shutil.copy2(input_file, run_directory / input_file.name)
    if request.param == "excel":
        for yaml_file in run_directory.glob("*.yaml"):
            yaml_file.unlink()
    return run_directory


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


def test_case_1_regression(case_1_run_directory):
    simulation = Simulation(str(case_1_run_directory))
    simulation.run()

    # The output tree is <run_directory>/<method>/<simulation name>/Output;
    # discover it instead of hard-coding the names from the input files.
    solution_directories = list(
        case_1_run_directory.glob("*/*/Output/Solution/CONDUCTOR_1")
    )
    assert len(solution_directories) == 1, (
        "expected exactly one simulation output tree, found: "
        f"{solution_directories}"
    )
    output_directory = solution_directories[0].parents[1]

    for reference_file in sorted((REFERENCE_DIRECTORY / "Solution").iterdir()):
        compare_against_reference(
            output_directory / "Solution" / "CONDUCTOR_1" / reference_file.name,
            reference_file,
        )
    for reference_file in sorted(
        (REFERENCE_DIRECTORY / "Time_evolution").iterdir()
    ):
        compare_against_reference(
            output_directory / "Time_evolution" / "CONDUCTOR_1" / reference_file.name,
            reference_file,
        )
