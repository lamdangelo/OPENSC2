"""Figures of the wp_model_no_transv run (run run_model.py first).

figures/            hot-spot time evolutions, spatial snapshots and the CEA
                    outlet-temperature comparison
publication-ready/  the publication set (detection voltage, outlet
                    temperatures vs measurement and THEA, flows, profiles,
                    hot spots)

Usage:
    .venv/bin/python models/w7x/wp_model_no_transv/make_figures.py [--model-dir DIR]
"""

import argparse
import sys
from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIRECTORY.parent))

from common import figure_library, publication_figures  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY,
                        help="run directory (default: this model; a smoke run: "
                        "smoke_runs/<run>)")
    run_directory = parser.parse_args().model_dir.resolve()
    figure_library.generate_for_model(run_directory, run_directory / "figures")
    publication_figures.generate(run_directory)
