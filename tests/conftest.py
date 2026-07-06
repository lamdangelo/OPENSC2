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
