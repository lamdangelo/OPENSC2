"""Driver of the formulation verification suite.

    python tests/verification/formulation_suite/driver.py --tier all --quick
    python tests/verification/formulation_suite/driver.py --tier 2 --formulation velocity --level h2
    python tests/verification/formulation_suite/driver.py --metrics --summary

Every run writes results/<tier>/<formulation>_<mode>_<level>/{state_snapshots.npz,
port_timeseries.csv, diagnostics.csv, run_manifest.json}; finished runs
(manifest present) are skipped unless --force. --metrics evaluates
results/<tier>/metrics.json from the files alone (no solver), --summary
writes results/summary.md. Figures: scripts/make_plots.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for entry in (REPOSITORY_ROOT / "source_code", REPOSITORY_ROOT / "tests",
              REPOSITORY_ROOT / "tests" / "verification"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from formulation_suite import tiers  # noqa: E402
from formulation_suite.heat_pulse import PEAK_LINEAR_POWER  # noqa: E402

TIERS = ("tier1", "tier2", "tier3a", "tier3b", "tier4", "tier5")
DEFAULT_RESULTS = REPOSITORY_ROOT / "results"


def tier_name(value: str) -> str:
    value = value.lower()
    return value if value.startswith("tier") else f"tier{value}"


def execute(spec: tiers.RunSpec, force: bool = False) -> dict:
    """Run one configuration (skipped when its manifest exists)."""
    from simulation import Simulation

    manifest_path = spec.run_directory / "run_manifest.json"
    if manifest_path.exists() and not force:
        print(f"[skip] {spec.tier}/{spec.name} (manifest exists)")
        return json.loads(manifest_path.read_text())
    lock = spec.run_directory / "running.lock"
    if lock.exists() and not force:
        print(f"[skip] {spec.tier}/{spec.name} (running elsewhere since {lock.read_text().strip()})")
        return {}
    run_directory = spec.run_directory / "run"
    if run_directory.exists():
        shutil.rmtree(run_directory)
    spec.run_directory.mkdir(parents=True, exist_ok=True)
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    spec.writer(run_directory)
    print(f"[run ] {spec.tier}/{spec.name}: elements {spec.parameters.get('elements')}, "
          f"dt {spec.parameters.get('time_step'):.3e} s", flush=True)
    start = time.perf_counter()
    simulation = Simulation(str(run_directory), step_callback=spec.recorder)
    simulation.run()
    spec.recorder.finish(simulation)
    wall = time.perf_counter() - start
    manifest = dict(spec.parameters)
    manifest.update({"mode": "single", "run_directory": str(run_directory),
                     "wall_time_total_s": wall, "steps_total": simulation.num_step,
                     "wall_time_per_step_s": wall / max(simulation.num_step, 1)})
    if spec.post is not None:
        manifest.update(spec.post(simulation) or {})
    spec.recorder.write(spec.run_directory, manifest)
    lock.unlink(missing_ok=True)
    # The solver's own output tree is large (figures); keep only the inputs.
    target = run_directory / "simulation_results"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    print(f"[done] {spec.tier}/{spec.name}: {simulation.num_step} steps, {wall:.1f} s wall, "
          f"settle {spec.recorder.settle_steps} + tier {spec.recorder.tier_steps} steps", flush=True)
    return json.loads((spec.run_directory / "run_manifest.json").read_text())


def calibrate_pulse(results_root: Path, target_peak_temperature: float = 18.0) -> float:
    """Coarse Tier-3a runs (velocity formulation): scale the pulse peak so
    that the peak strand temperature reaches the target."""
    from simulation import Simulation

    peak = PEAK_LINEAR_POWER
    history = []
    for iteration in range(4):
        spec = tiers.tier3a_spec("velocity", "h", results_root / "calibration", peak=peak,
                                 duration=0.4)
        run_directory = spec.run_directory / f"run_{iteration}"
        if run_directory.exists():
            shutil.rmtree(run_directory)
        spec.writer(run_directory)
        simulation = Simulation(str(run_directory), step_callback=spec.recorder)
        simulation.run()
        spec.recorder.finish(simulation)
        peak_T = max(v[0] for v in spec.recorder.max_strand_temperature.values())
        history.append({"peak_linear_power": peak, "peak_strand_temperature": peak_T})
        print(f"[calib] peak {peak:.4e} W/m -> T_max {peak_T:.3f} K", flush=True)
        shutil.rmtree(run_directory, ignore_errors=True)
        if abs(peak_T - target_peak_temperature) < 0.3:
            break
        rise = max(peak_T - tiers.geometry.TEMPERATURE, 1e-3)
        peak *= (target_peak_temperature - tiers.geometry.TEMPERATURE) / rise
    (results_root / "pulse_calibration.json").write_text(json.dumps(history, indent=1))
    print(f"[calib] set heat_pulse.PEAK_LINEAR_POWER = {peak:.6e}")
    return peak


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="all", help="1, 2, 3a, 3b, 4 or all (comma list)")
    parser.add_argument("--formulation", default="both", choices=("velocity", "mass_flow", "both"))
    parser.add_argument("--mode", default="single", choices=("single",),
                        help="only single-pass exists (no Picard sub-iteration in the solver)")
    parser.add_argument("--level", default="all", help="h, h2, h4, h8 or all (comma list)")
    parser.add_argument("--quick", action="store_true", help="coarsest level only, no series variants")
    parser.add_argument("--variant", default="all", help="run only this variant (prop, htc1, dx, dt) or all")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--calibrate-pulse", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="evaluate metrics.json from results/")
    parser.add_argument("--summary", action="store_true", help="write results/summary.md")
    parser.add_argument("--no-run", action="store_true", help="skip the solver runs")
    args = parser.parse_args(argv)

    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    tiers.save_operating_point(results_root)
    if args.calibrate_pulse:
        calibrate_pulse(results_root)
        return 0

    selected_tiers = TIERS if args.tier == "all" else [tier_name(t) for t in args.tier.split(",")]
    formulations = tiers.FORMULATIONS if args.formulation == "both" else (args.formulation,)
    levels = ["h", "h2", "h4"] if args.level == "all" else args.level.split(",")

    if not args.no_run:
        for tier in selected_tiers:
            for spec in tiers.run_specs(tier, formulations, levels, results_root, quick=args.quick):
                if args.variant != "all" and spec.variant != args.variant:
                    continue
                execute(spec, force=args.force)
    if args.metrics or args.summary:
        from formulation_suite import metrics as metrics_module
        for tier in selected_tiers:
            if (results_root / tier).exists():
                metrics_module.evaluate_tier(tier, results_root)
    if args.summary:
        from formulation_suite import summary as summary_module
        summary_module.write_summary(results_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
