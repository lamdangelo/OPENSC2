"""Regression test for TDD example 2 (CASE_2_ENEA_HTS_CICC).

The example as shipped simulates 50 s with a 15--16 s heat pulse on
STR_MIX_1 and takes about half an hour — far too long for the CI pipeline.
The test therefore runs a shortened scenario, edited into a temporary copy
of the YAML input files:

* ``end_time`` reduced to 2 s;
* the heat pulse moved to 0.5--1.5 s (same amplitude, extent and duration),
  so the transient still covers the pre-pulse steady state, the full pulse
  and the early temperature response of all 38 components;
* the spatial-distribution save times trimmed to the shortened window.

Only the YAML input format is exercised: shortening the Excel variant would
require rewriting workbooks that contain formula cells, which openpyxl
cannot save without corrupting their cached values. Format equivalence is
guarded by the CASE_1 regression test (both formats) and was verified for
CASE_2 against a full-length run when the reference data was generated.

See ``regression_utilities`` for the comparison strategy and how the
committed reference data in ``tests/reference_data/CASE_2_ENEA_HTS_CICC``
is regenerated (use this module's ``shorten_scenario`` on a copy of the
inputs to reproduce the exact test scenario).
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
CASE_2_INPUT_DIRECTORY = REPOSITORY_ROOT / "TDD_examples" / "CASE_2_ENEA_HTS_CICC"
REFERENCE_DIRECTORY = Path(__file__).parent / "reference_data" / "CASE_2_ENEA_HTS_CICC"

SHORTENED_END_TIME = 2.0
SHORTENED_HEAT_PULSE_START = 0.5
SHORTENED_HEAT_PULSE_END = 1.5
SHORTENED_SPATIAL_SAVE_TIMES = [1.0]


def shorten_scenario(run_directory: Path) -> None:
    """Edit the copied YAML inputs into the shortened CI scenario."""
    simulation_file = run_directory / "simulation.yaml"
    simulation_document = yaml.safe_load(simulation_file.read_text())
    simulation_document["simulation"]["end_time"] = SHORTENED_END_TIME
    simulation_file.write_text(yaml.safe_dump(simulation_document, sort_keys=False))

    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    conductor_document = yaml.safe_load(conductor_file.read_text())
    conductor_document["diagnostics"]["spatial_distribution_times"] = (
        SHORTENED_SPATIAL_SAVE_TIMES
    )
    for component in conductor_document["components"]:
        operations = component.get("operations", {})
        if operations.get("heat_flux_mode"):
            operations["heat_flux_time_start"] = SHORTENED_HEAT_PULSE_START
            operations["heat_flux_time_end"] = SHORTENED_HEAT_PULSE_END
    conductor_file.write_text(yaml.safe_dump(conductor_document, sort_keys=False))


@pytest.fixture
def case_2_run_directory(tmp_path):
    run_directory = tmp_path / "CASE_2_ENEA_HTS_CICC"
    run_directory.mkdir()
    copy_input_files(CASE_2_INPUT_DIRECTORY, run_directory, "yaml")
    shorten_scenario(run_directory)
    return run_directory


def test_case_2_regression(case_2_run_directory):
    simulation = Simulation(str(case_2_run_directory))
    simulation.run()

    output_directory = discover_output_directory(case_2_run_directory)
    compare_output_tree(output_directory, REFERENCE_DIRECTORY)
