"""Run the adiabatic (single-conductor, 1D) W7-X model.

One W7-X double-layer conductor at 4.1 K / 15320 A with a 2 J heat impulse
at 80.0-80.1 m, prescribed current decay (external_current.csv), no hydraulic
network, no transverse conduction, no induced losses: the OPENSC2 counterpart
of the THEA reference run (validation_data/thea/high_fidelity_data_*.csv).

Usage (from the repository root, with the repository .venv):
    .venv/bin/python models/w7x/adiabatic_model/run_model.py [--mesh-elements N --end-time T]

Outputs: simulation_results/W7X/ in this directory; make_figures.py draws
the hot-spot comparison with THEA from them.
"""

import sys
from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIRECTORY.parent))

from common.run_wp_model import run_plain  # noqa: E402

if __name__ == "__main__":
    run_plain(MODEL_DIRECTORY, description=__doc__)
