"""Figures for a run WITHOUT hydraulic network (no network history file).

Standard set (figures/): the per-DL hot-spot panels and the spatial
overlays of generate_figures.py (the network overlay panels are skipped).
Publication set (publication-ready/): the network-independent figures of
publication_figures.py (detection voltage, strand hot spots, DL inlet and
outlet flows, strand temperature profiles at several snapshots) plus the
channel outlet temperature of every DL, which replaces the outlet-group
temperatures of the network runs.

Usage:
    python -m common.no_network_figures --model-dir DIR   (from models/w7x)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import figure_library as gf  # noqa: E402
from . import publication_figures as pf  # noqa: E402

PROFILE_TIMES = (2.0, 5.0, 10.0)  # event seconds


def dl_outlet_temperatures(model_directory, alignment, output_directory):
    """Channel outlet temperature of every DL over the event (same size as
    the narrow outlet-group temperature figures of the network runs)."""
    window = (-1.0, 15.0)
    time_evolution = gf.output_directory_of(model_directory) / "Time_evolution"
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for conductor, marker in zip(gf.CONDUCTORS, pf.DL_MARKERS):
        frame = pd.read_csv(
            gf.component_file(
                time_evolution / f"CONDUCTOR_{conductor}", "CHAN1",
                "inlet_outlet_te.tsv",
            ),
            sep="\t",
        )
        time = gf.to_event_time(frame["time (s)"].to_numpy(), alignment)
        model_t, temperature = gf.in_window(
            time, frame["temperature_out (K)"].to_numpy(), window
        )
        double_layer = gf.DOUBLE_LAYER[conductor]
        axis.plot(model_t, temperature, "-", lw=2.0,
                  color=pf.DL_VIRIDIS[double_layer - 1], marker=marker,
                  markevery=0.05, markersize=6, label=f"DL{double_layer}")
    axis.set_ylabel("outlet temperature (K)")
    pf.event_axis(axis, window)
    pf.legend_below(figure, ncol=6)
    return pf.save(figure, output_directory, "dl_outlet_temperatures")


def generate(model_directory: Path, figures_dir: Path | None = None,
             publication_dir: Path | None = None) -> None:
    """Standard and publication figure sets of a run without network."""
    model_directory = Path(model_directory)
    log = pd.read_csv(model_directory / "final_run_log.tsv", sep="\t")
    alignment = gf.quench_alignment(log)
    hotspot_log = pd.read_csv(model_directory / "hotspot_log.tsv", sep="\t")
    output_root = gf.output_directory_of(model_directory)

    # Standard set.
    figures = figures_dir or model_directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pre_time, post_time = gf.quench_snapshots(output_root)
    panels: dict = {}
    panels.update(gf.hotspot_panels(output_root, alignment, hotspot_log))
    panels.update(gf.spatial_panels(output_root, alignment, pre_time, post_time))
    written = gf.save_panels(panels, figures)
    print(f"{model_directory.name}: wrote {len(written)} EPS figures to {figures}")

    # Publication set.
    plt.rcParams["font.size"] = pf.BASE_FONT
    publication = publication_dir or model_directory / "publication-ready"
    publication.mkdir(parents=True, exist_ok=True)
    log_time = gf.to_event_time(log["time"].to_numpy(), alignment)
    pf.detection_voltage(log, log_time, publication)
    pf.strand_hotspots(hotspot_log, alignment, publication)
    pf.dl_end_flows(model_directory, alignment, publication, end="inlet")
    pf.dl_end_flows(model_directory, alignment, publication, end="outlet")
    pf.dl_flows_combined(model_directory, alignment, publication)
    dl_outlet_temperatures(model_directory, alignment, publication)
    for event_time in PROFILE_TIMES:
        for profile in (pf.strand_profiles, pf.channel_pressure_profiles):
            profile(model_directory, alignment, event_time, publication)
            profile(model_directory, alignment, event_time, publication,
                    width=6.4, suffix="_narrow")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--publication-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    generate(arguments.model_dir, arguments.figures_dir, arguments.publication_dir)


if __name__ == "__main__":
    main()
