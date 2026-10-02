"""Figures of the wp_model_no_network run (run run_model.py first).

figures/            hot-spot time evolutions and spatial snapshots
publication-ready/  detection voltage, strand hot spots, DL flows, DL outlet
                    temperatures, strand and channel-pressure profiles

Usage:
    .venv/bin/python models/w7x/wp_model_no_network/make_figures.py [--model-dir DIR]
"""

import argparse
import sys
from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIRECTORY.parent))

from common import no_network_figures  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY,
                        help="run directory (default: this model; a smoke run: "
                        "smoke_runs/<run>)")
    no_network_figures.generate(parser.parse_args().model_dir.resolve())
