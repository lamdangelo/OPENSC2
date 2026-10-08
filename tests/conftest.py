"""Shared pytest configuration for the OPENSC2 test suite."""

import os

import matplotlib

# The simulation produces figures as a side effect; force the non-interactive
# Agg backend so the tests run on headless machines (e.g. CI runners) exactly
# as they do on a workstation. Setting MPLBACKEND (not just matplotlib.use)
# matters: utility_functions/plots.py selects the GUI backend at import time
# unless an explicit backend was requested through the environment. Must
# happen before any pyplot import.
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# Markers (see [tool.pytest.ini_options] in pyproject.toml). Tests that run a
# full Simulation are marked by file so the developer loop can deselect them
# with ``pytest -m "not slow"``; the CI runs everything.
# ---------------------------------------------------------------------------

SLOW_TEST_FILES = {
    "test_case1_regression.py",
    "test_case2_regression.py",
    "test_case3_regression.py",
    "test_checkpoint_restart.py",
    "test_mass_flow_explicit_integration.py",
    "test_network_coupling.py",
    "test_conductor_mesh.py",
}
REGRESSION_TEST_FILES = {
    "test_case1_regression.py",
    "test_case2_regression.py",
    "test_case3_regression.py",
}


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        name = item.path.name
        if "verification" in item.path.parts:
            item.add_marker(pytest.mark.verification)
            item.add_marker(pytest.mark.slow)
        if name in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
        if name in REGRESSION_TEST_FILES:
            item.add_marker(pytest.mark.regression)
