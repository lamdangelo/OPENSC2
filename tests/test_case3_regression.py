"""Regression test for TDD example 3 (CASE_3_HTS_HVDC).

The example as shipped simulates 20 000 s of a nitrogen-cooled HVDC cable
approaching its steady state under environmental parasitic load (2 000 fixed
10 s steps) — several minutes of wall time plus post-processing, too long to
run twice in the CI pipeline. The test therefore runs a shortened scenario,
edited into a temporary copy of the YAML input files:

* ``end_time`` reduced to 2 000 s (200 steps) — the channel temperature has
  then completed most of its rise towards the steady state (31.17 K of the
  final 31.29 K, starting from 30 K), so the transient physics is covered;
* the spatial-distribution save times trimmed to the shortened window
  (there is no heat pulse to relocate in this example).

Only the YAML input format is exercised: shortening the Excel variant would
require rewriting workbooks that contain formula cells, which openpyxl
cannot save without corrupting their cached values. Format equivalence is
guarded by the CASE_1 regression test (both formats) and was verified for
CASE_3 against a full-length run when the reference data was generated
(bit-identical outputs).

See ``regression_utilities`` for the comparison strategy and how the
committed reference data in ``tests/reference_data/CASE_3_HTS_HVDC`` is
regenerated (use this module's ``shorten_scenario`` on a copy of the inputs
to reproduce the exact test scenario).
"""

from pathlib import Path

import pytest
import yaml

from simulation import Simulation

from regression_utilities import (
    compare_output_tree,
    copy_input_files,
    discover_output_directory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_3_INPUT_DIRECTORY = REPOSITORY_ROOT / "TDD_examples" / "CASE_3_HTS_HVDC"
REFERENCE_DIRECTORY = Path(__file__).parent / "reference_data" / "CASE_3_HTS_HVDC"

SHORTENED_END_TIME = 2000.0


def shorten_scenario(run_directory: Path) -> None:
    """Edit the copied YAML inputs into the shortened CI scenario."""
    simulation_file = run_directory / "simulation.yaml"
    simulation_document = yaml.safe_load(simulation_file.read_text())
    simulation_document["simulation"]["end_time"] = SHORTENED_END_TIME
    simulation_file.write_text(yaml.safe_dump(simulation_document, sort_keys=False))

    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    conductor_document = yaml.safe_load(conductor_file.read_text())
    diagnostics = conductor_document["diagnostics"]
    diagnostics["spatial_distribution_times"] = [
        save_time
        for save_time in diagnostics["spatial_distribution_times"]
        if save_time <= SHORTENED_END_TIME
    ]
    conductor_file.write_text(yaml.safe_dump(conductor_document, sort_keys=False))


@pytest.fixture
def case_3_run_directory(tmp_path):
    run_directory = tmp_path / "CASE_3_HTS_HVDC"
    run_directory.mkdir()
    copy_input_files(CASE_3_INPUT_DIRECTORY, run_directory, "yaml")
    shorten_scenario(run_directory)
    return run_directory


def test_case_3_regression(case_3_run_directory):
    simulation = Simulation(str(case_3_run_directory))
    simulation.run()

    output_directory = discover_output_directory(case_3_run_directory)
    compare_output_tree(output_directory, REFERENCE_DIRECTORY)
