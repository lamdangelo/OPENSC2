"""Run the wp_model_no_network model.

Winding-pack model WITHOUT hydraulic network: six double layers with imposed channel pressure drops, transverse conduction and induced losses (58 ms); quench started by a 2 J heat impulse on DL1; 15 s simulated after the dump.

Usage (from the repository root, with the repository .venv):
    .venv/bin/python models/w7x/wp_model_no_network/run_model.py [--help] [--mesh-elements N --end-time T]

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
    main(MODEL_DIRECTORY, coupling_tau=58e-3, apply_schedules=False, post_discharge=15.0, description=__doc__)
