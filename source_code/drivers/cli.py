"""Headless command-line driver.

    python -m drivers.cli <model_directory>

Runs the simulation defined by the input files (YAML or legacy Excel) in
``<model_directory>`` and writes the results next to them, exactly as the
model scripts in ``models/`` do, without any ``sys.path`` manipulation.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an OPENSC2 simulation headless.")
    parser.add_argument("model_directory", help="directory containing the input files")
    arguments = parser.parse_args(argv)
    # Headless by default; an explicit MPLBACKEND wins.
    os.environ.setdefault("MPLBACKEND", "Agg")

    from simulation import Simulation

    directory = os.path.abspath(arguments.model_directory)
    if not os.path.isdir(directory):
        parser.error(f"not a directory: {directory}")
    Simulation(directory + os.sep).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
