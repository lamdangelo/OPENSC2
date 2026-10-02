"""Run the wp_model_no_induced_losses model.

Winding-pack model of the AAB51 quench test WITHOUT induced losses (coupling, eddy and hysteresis losses off); network, transverse conduction and schedules as in the full-feature model.

Usage (from the repository root, with the repository .venv):
    .venv/bin/python models/w7x/wp_model_no_induced_losses/run_model.py [--help] [--mesh-elements N --end-time T]

Outputs: simulation_results/W7X_QT/ (OPENSC2), final_run_log.tsv and
hotspot_log.tsv (controller) in this directory; make_figures.py draws from
them.
"""

import sys
from pathlib import Path

MODEL_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIRECTORY.parent))

from common.run_wp_model import main  # noqa: E402

if __name__ == "__main__":
    main(MODEL_DIRECTORY, induced_losses=False, description=__doc__)
