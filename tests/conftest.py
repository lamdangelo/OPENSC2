"""Shared pytest configuration for the OPENSC2 test suite."""

import matplotlib

# The simulation produces figures as a side effect; force the non-interactive
# Agg backend so the tests run on headless machines (e.g. CI runners) exactly
# as they do on a workstation. Must happen before any pyplot import.
matplotlib.use("Agg", force=True)
