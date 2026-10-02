"""Command line entry point shared by the ``run_model.py`` of every model.

``main`` runs a winding-pack (WP) model under the quench-test controller,
``run_plain`` runs a model without controller (the adiabatic 1D model, whose
current is fully prescribed). Both accept the smoke-test options
``--mesh-elements`` and ``--end-time``: they never modify the tracked deck
but run a reduced copy of it under ``<model>/smoke_runs/``.

Each model's ``run_model.py`` calls ``main`` with its own defaults baked in;
the defaults are exposed as the argparse defaults so that
``run_model.py --help`` shows the configuration of that model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

from .constants import POST_DISCHARGE_DURATION

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CODE = REPOSITORY_ROOT / "source_code"

os.environ.setdefault("MPLBACKEND", "Agg")  # never open figure windows
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))


def prepare_scratch_deck(model_directory: Path, mesh_elements, end_time) -> Path:
    """Copy the deck to ``<model>/smoke_runs/mesh<N>_tend<T>/`` with the
    requested mesh size and end time, leaving the tracked deck untouched.

    The number of elements is written to every conductor
    (``maximum_number_of_nodes`` = elements + 50 as in the decks; a refined
    mesh becomes a uniform one of that size) and the spatial snapshot times
    beyond the new end time are dropped (OPENSC2 refuses snapshots after
    TEND).
    """
    tag = f"mesh{mesh_elements or 'deck'}_tend{end_time or 'deck'}"
    scratch = model_directory / "smoke_runs" / tag
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for source in model_directory.iterdir():
        if source.suffix in (".yaml", ".csv") and source.is_file():
            shutil.copy2(source, scratch / source.name)

    simulation_file = scratch / "simulation.yaml"
    document = yaml.safe_load(simulation_file.read_text())
    if end_time is not None:
        document["simulation"]["end_time"] = float(end_time)
    simulation_file.write_text(yaml.safe_dump(document, sort_keys=False))

    for conductor_file in sorted(scratch.glob("conductor_*.yaml")):
        document = yaml.safe_load(conductor_file.read_text())
        grid = document["grid"]
        if mesh_elements is not None:
            grid["number_of_elements"] = int(mesh_elements)
            grid["maximum_number_of_nodes"] = int(mesh_elements) + 50
            if int(grid.get("mesh_type", 0)) == 1:
                # A reduced refined mesh rarely satisfies the mesh validator
                # (pitch and growth constraints); the smoke deck uses a
                # uniform mesh of the requested size instead.
                grid["mesh_type"] = 0
        if end_time is not None:
            diagnostics = document.get("diagnostics", {})
            times = diagnostics.get("spatial_distribution_times")
            if times:
                kept = [t for t in times if t <= float(end_time)]
                diagnostics["spatial_distribution_times"] = kept or [float(end_time)]
        conductor_file.write_text(yaml.safe_dump(document, sort_keys=False))
    print(f"[driver] reduced deck written to {scratch}")
    return scratch


def _add_smoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh-elements", type=int, default=None,
        help="smoke test: run a copy of the deck with this many elements per "
        "conductor (under <model>/smoke_runs/); the deck is not modified",
    )
    parser.add_argument(
        "--end-time", type=float, default=None,
        help="smoke test: end time (s) of the reduced deck copy",
    )


def _resolve_run_directory(arguments) -> Path:
    model_directory = Path(arguments.model_dir).resolve()
    if arguments.mesh_elements is None and arguments.end_time is None:
        return model_directory
    return prepare_scratch_deck(
        model_directory, arguments.mesh_elements, arguments.end_time
    )


def main(model_directory: Path, *, coupling_tau: float = 58e-3,
         induced_losses: bool = True, apply_schedules: bool = True,
         post_discharge: float = POST_DISCHARGE_DURATION,
         description: str | None = None, argv=None) -> None:
    """Run a WP model under the quench-test controller.

    The keyword arguments are the model's configuration; they become the
    defaults of the command line options so that each ``run_model.py``
    reproduces its paper run without arguments.
    """
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path(model_directory))
    parser.add_argument(
        "--no-schedules", action="store_true", default=not apply_schedules,
        help="do not drive the supply reservoir with the measured "
        "PT2111/TT2113C traces (runs without hydraulic network)",
    )
    parser.add_argument(
        "--post-discharge", type=float, default=post_discharge,
        help="seconds simulated past the start of the current dump",
    )
    parser.add_argument(
        "--coupling-tau", type=float, default=coupling_tau,
        help="coupling-loss coefficient n*tau (s) applied to every strand",
    )
    parser.add_argument(
        "--no-induced-losses", action="store_true", default=not induced_losses,
        help="zero coupling, eddy and hysteresis losses on every strand and "
        "jacket (takes precedence over --coupling-tau)",
    )
    _add_smoke_arguments(parser)
    arguments = parser.parse_args(argv)
    run_directory = _resolve_run_directory(arguments)

    from simulation import Simulation  # noqa: E402

    from .controller import QuenchTestController

    controller = QuenchTestController(
        run_directory,
        apply_schedules=not arguments.no_schedules,
        post_discharge_duration=arguments.post_discharge,
        induced_losses=not arguments.no_induced_losses,
        coupling_tau=arguments.coupling_tau,
    )
    simulation = Simulation(str(run_directory) + "/", step_callback=controller)
    try:
        simulation.run()
    finally:
        controller.flush_log()
    if controller.detection_time is not None:
        from .constants import SIMULATION_START_MEASURED_TIME

        measured = controller.detection_time + SIMULATION_START_MEASURED_TIME
        print(
            f"detection at t_sim = {controller.detection_time:.4f} s "
            f"(measured time {measured:+.1f} s; experiment: +0.461 s)"
        )
    else:
        print("no detection: no double layer reached 0.5 V")


def run_plain(model_directory: Path, *, description: str | None = None,
              argv=None) -> None:
    """Run a model without controller (prescribed current, no schedules)."""
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path(model_directory))
    _add_smoke_arguments(parser)
    arguments = parser.parse_args(argv)
    run_directory = _resolve_run_directory(arguments)

    from simulation import Simulation  # noqa: E402

    Simulation(str(run_directory) + "/").run()
