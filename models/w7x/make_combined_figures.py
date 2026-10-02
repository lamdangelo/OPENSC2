"""Combined 2 x 2 publication figures of the four winding-pack model variants.

Panels: a) full-feature run (top left), b) no induced losses (top right),
c) no hydraulic network (bottom left), d) no transverse heat conduction
(bottom right); one legend at the bottom of the figure; the per-run style
rules otherwise (units in round brackets, "event time (s)", fonts 1.25 x the
matplotlib default; panel titles centred, another 1.25 x larger; a gap
between the rows). The panels reuse the axis-level drawing functions of
``common/publication_figures.py``.

All four runs must have been executed (run_model.py of each model); the
figures are written to ``publication-ready/`` next to this script.

Usage (from models/w7x):
    ../../.venv/bin/python make_combined_figures.py
        [--root DIR] [--dirs FULL NOLOSS NONET NOTC] [--profile-dir NAME=TWIN]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STUDY = Path(__file__).resolve().parent
OUTPUT = STUDY / "publication-ready"

# (directory, panel title).
VARIANTS = (
    ("wp_model_full", "a) full-feature run"),
    ("wp_model_no_induced_losses", "b) no induced losses"),
    ("wp_model_no_network", "c) no hydraulic network"),
    ("wp_model_no_transv", "d) no transverse heat conduction"),
)
# A run whose strand-profile panel is taken from a twin run with other
# snapshot times (name -> twin name); empty for the packaged models, whose
# decks save the +2 s snapshot themselves.
PROFILE_DIRECTORIES: dict = {}

sys.path.insert(0, str(STUDY))
from common import publication_figures as pf  # noqa: E402

gf = pf.gf
PANEL_SIZE = (6.4, 4.6)  # inches, one panel (the per-run narrow width)
TITLE_FONT = pf.BASE_FONT * 1.25   # 15.6 pt
LEGEND_FONT = pf.LEGEND_FONT * 1.25  # 12.5 pt
# Outlet groups of the runs without the hydraulic network, approximated
# from the channel outlet temperatures of the double layers feeding them.
APPROXIMATE_OUTLET_GROUPS = {2: (2, 3), 3: (4, 5), 4: (6,)}
ROW_GAP = 0.08                     # fraction of panel height between the rows


class RunData:
    """Lazy per-run data access with the per-run event-time alignment."""

    def __init__(self, name: str):
        self.directory = STUDY / name
        self.log = pd.read_csv(self.directory / "final_run_log.tsv", sep="\t")
        self.alignment = gf.quench_alignment(self.log)
        self.log_time = gf.to_event_time(self.log["time"].to_numpy(),
                                         self.alignment)
        self.hotspot_log = pd.read_csv(self.directory / "hotspot_log.tsv",
                                       sep="\t")
        try:
            self.network = gf.read_network_history(self.directory)
            self.model_time = gf.to_event_time(
                self.network["time (s)"].to_numpy(), self.alignment
            )
        except FileNotFoundError:
            self.network, self.model_time = None, None

    def channel_outlet_temperature(self, double_layer: int):
        """(event time, outlet temperature) of one DL channel."""
        output = gf.output_directory_of(self.directory)
        frame = pd.read_csv(
            output / "Time_evolution" / f"CONDUCTOR_{100 * double_layer}"
            / f"CHAN1_D{double_layer}_inlet_outlet_te.tsv", sep="\t",
        )
        return (gf.to_event_time(frame["time (s)"].to_numpy(), self.alignment),
                frame["temperature_out (K)"].to_numpy())


def combined_figure(stem: str, draw, ncol: int, height_scale: float = 1.0,
                    legend_from: int = 0, legend_key=None) -> Path:
    """2 x 2 panels, ``draw(axis, run, name)`` per variant, one legend
    (handles of panel ``legend_from``, or the handles ``draw`` returns;
    entries sorted by ``legend_key(label)`` when given; matplotlib fills
    the legend column by column)."""
    figure, axes = plt.subplots(
        2, 2, figsize=(2 * PANEL_SIZE[0], 2 * PANEL_SIZE[1] * height_scale),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(hspace=ROW_GAP)
    handles, labels = [], []
    for index, ((name, title), axis) in enumerate(zip(VARIANTS, axes.flat)):
        run = RunData(name)
        returned = draw(axis, run, name)
        axis.set_title(title, fontsize=TITLE_FONT, loc="center")
        if index == legend_from:
            if returned is not None:
                handles, labels = list(returned[0]), list(returned[1])
            else:
                panel_handles, panel_labels = axis.get_legend_handles_labels()
                handles, labels = list(panel_handles), list(panel_labels)
        else:
            # Entries that only this panel draws (e.g. approximated outlets).
            for handle, label in zip(*axis.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    if legend_key is not None:
        order = sorted(range(len(labels)), key=lambda i: legend_key(labels[i]))
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    figure.legend(handles, labels, loc="outside lower center", ncol=ncol,
                  fontsize=LEGEND_FONT, frameon=True)
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / f"{stem}.eps"
    figure.savefig(target, format="eps")
    plt.close(figure)
    print(f"[figure] {target}")
    return target


def detection_voltage(axis, run, name):
    return pf.draw_detection_voltage(axis, run.log, run.log_time, smoothed=True)


def strand_hotspots(axis, run, name):
    pf.draw_strand_hotspots(axis, run.hotspot_log, run.alignment)


def outlet_groups_234(axis, run, name):
    pf.draw_cea_groups_234(axis, run.network, run.model_time,
                           gf.CEA_PLOT_WINDOW, legend=False)
    if run.network is None:
        # No outlet headers: approximate each group by the (mean) channel
        # outlet temperature of the double layers that feed it.
        window = gf.CEA_PLOT_WINDOW
        for group, double_layers in APPROXIMATE_OUTLET_GROUPS.items():
            traces = [run.channel_outlet_temperature(n) for n in double_layers]
            time = traces[0][0]
            temperature = np.mean([t for _, t in traces], axis=0)
            model_t, model_temperature = gf.in_window(time, temperature, window)
            axis.plot(model_t, model_temperature, "-",
                      color=gf.GROUP_COLOURS[group], marker="^",
                      markevery=0.05, markersize=6,
                      label=f"Outlet {group}, approx.")


def outlet_legend_key(label: str):
    """One legend column per kind: WP | meas. | approx., groups in order."""
    kind = label.split(", ")[1]
    return (("WP", "meas.", "approx.").index(kind), label)


def strand_profiles(axis, run, name):
    profile_run = run
    profile_name = PROFILE_DIRECTORIES.get(name)
    if profile_name is not None:
        profile_run = RunData(profile_name)
    saved = pf.draw_spatial_profiles(
        axis, profile_run.directory, profile_run.alignment, 2.0,
        "STR_MIX1", "temperature", "strand temperature (K)",
    )
    print(f"[snapshot] {profile_run.directory.name}: requested +2.0 s, "
          f"saved {saved:+.2f} s")


def main(argv=None) -> None:
    global STUDY, OUTPUT, VARIANTS, PROFILE_DIRECTORIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STUDY,
                        help="directory holding the four run directories")
    parser.add_argument("--dirs", nargs=4, default=None,
                        metavar=("FULL", "NOLOSS", "NONET", "NOTC"),
                        help="run directory names in panel order")
    parser.add_argument("--profile-dir", action="append", default=[],
                        metavar="NAME=TWIN",
                        help="take the strand-profile panel of NAME from TWIN")
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    STUDY = Path(arguments.root).resolve()
    OUTPUT = arguments.output_dir or (Path(__file__).resolve().parent / "publication-ready")
    if arguments.dirs is not None:
        VARIANTS = tuple(
            (name, title) for name, (_, title) in zip(arguments.dirs, VARIANTS)
        )
    for item in arguments.profile_dir:
        name, twin = item.split("=", 1)
        PROFILE_DIRECTORIES[name] = twin

    plt.rcParams["font.size"] = pf.BASE_FONT
    combined_figure("combined_detection_voltage_smoothed", detection_voltage,
                    ncol=7)
    combined_figure("combined_strand_hotspot_all_dl", strand_hotspots, ncol=6)
    combined_figure("combined_outlet_temperatures_cea_groups_2-4",
                    outlet_groups_234, ncol=3, legend_key=outlet_legend_key)
    combined_figure("combined_strand_temperature_profiles_2.0s",
                    strand_profiles, ncol=6)


if __name__ == "__main__":
    main()
