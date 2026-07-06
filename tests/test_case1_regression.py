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
two input paths. See ``regression_utilities`` for the comparison strategy
and how to regenerate the reference data.
"""

from pathlib import Path

import pytest

from simulation import Simulation

from regression_utilities import (
    compare_output_tree,
    copy_input_files,
    discover_output_directory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_1_INPUT_DIRECTORY = REPOSITORY_ROOT / "TDD_examples" / "CASE_1_ITER_like_LTS"
REFERENCE_DIRECTORY = Path(__file__).parent / "reference_data" / "CASE_1_ITER_like_LTS"


@pytest.fixture(params=["yaml", "excel"])
def case_1_run_directory(request, tmp_path):
    run_directory = tmp_path / "CASE_1_ITER_like_LTS"
    run_directory.mkdir()
    copy_input_files(CASE_1_INPUT_DIRECTORY, run_directory, request.param)
    return run_directory


def test_case_1_regression(case_1_run_directory):
    simulation = Simulation(str(case_1_run_directory))
    simulation.run()

    output_directory = discover_output_directory(case_1_run_directory)
    compare_output_tree(output_directory, REFERENCE_DIRECTORY)
