"""Standard EPS figures of one winding-pack (WP) run.

All figures are written as vector EPS files, one plot per file, into the
run's ``figures`` directory. Three groups are produced:

1. Hot-spot (maximum) temperature time evolution for each of the six
   conductors (DL1..DL6), with the strand (STR_MIX), jacket (Z_JACKET1)
   and epoxy (Z_JACKET2) on one axis, from the controller's hot-spot log
   (true spatial maximum) when present.

2. Strand (STR_MIX) temperature vs. position, all six conductors overlaid,
   at the last saved snapshot (after quench) and at the last snapshot before
   thermal runaway. Both snapshot times are detected from the run.

3. Channel (CHAN) pressure, velocity and mass flow rate vs. position, all
   six conductors overlaid, at the same two snapshots as (2).

plus the CEA outlet-temperature comparison (measured group outlets, 1D THEA
run, this run). The event-time alignment and the two snapshots are detected
per run, so the same module serves every variant. A run that never quenches
is placed on the nominal measured schedule and its zoomed close-ups are
skipped. The module also holds the data-access helpers used by
``publication_figures`` and ``no_network_figures``.

Usage (from models/w7x):
    ../../.venv/bin/python -m common.figure_library --model-dir DIR [--output-dir figures]
"""

from __future__ import annotations

import argparse
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VALIDATION_DATA_ROOT = Path(__file__).resolve().parents[1] / "validation_data"

from . import measurement_data as md  # noqa: E402
from .constants import (  # noqa: E402
    MEASURED_DETECTION_TIME,
    SIMULATION_START_MEASURED_TIME,
)

MBAR = 100.0
BAR = 1.0e5

GROUP_TEMPERATURE_SENSOR = {1: "TT2100A", 2: "TT2101A", 3: "TT2102A",
                            4: "TT2103A"}
# Shared pair-headers of the winding inlet side: pair i feeds DL(2i-1)
# and DL(2i) (ports in simulation.yaml).
PAIR_HEADER_DOUBLE_LAYERS = {1: (1, 2), 2: (3, 4), 3: (5, 6)}
PAIR_COLOURS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
# One colour per group for the combined outlet-temperature figure (the
# measured/computed pair of a group shares its colour).
GROUP_COLOURS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}

# MPG corporate colours and styles for the CEA-measurement comparison figure
# (same 1D/2D convention as W7-X/plot_hotspot_comparison.py: 1D in
# MPG_green_light {198,211,37} with diamond markers, 2D in MPG_green with
# circles).
MPG_GREEN = (0 / 255, 108 / 255, 102 / 255)
MPG_GREEN_LIGHT = (198 / 255, 211 / 255, 37 / 255)
MPG_BLUE_LIGHT = (0 / 255, 177 / 255, 234 / 255)
# 1D THEA run of the CEA test (already on the measured event-time axis).
THEA_ONE_D_FILE = VALIDATION_DATA_ROOT / "thea" / "cea_data_thermal.csv"
CEA_FONT_SCALE = 1.25
CEA_PLOT_WINDOW = (-5.0, 15.0)

# Model DL colours (matplotlib default cycle), reused so a measured tap can
# borrow the colour of its paired DL in a lighter tone.
DL_COLOURS = {double_layer: f"C{double_layer - 1}" for double_layer in range(1, 7)}

# Measured detection-voltage tap (DUP11 bridge, compensated across the
# double-layer pair (1,2)-(3,4); clips at 1 V).
DETECTION_SENSOR = "DUP11-3"
DETECTION_LIGHTEN = 0.55  # fraction of the way to white for measured taps
PLOT_WINDOW = (-5.0, 9.0)
# "Expanded" overlay variants: start at -2 s and run to the end of the
# available post-quench data (capped here), with the y-axis rescaled to the
# visible data. The upper bound is clipped to each run's actual data extent
# at figure time, so short (~9 s) runs stay ~9 s and long (30 s post-dump)
# runs show the full tail.
EXPANDED_WINDOW = (-2.0, 30.0)

# Conductors carry a "double layer" label DL1..DL6 in the report.
CONDUCTORS = (100, 200, 300, 400, 500, 600)
DOUBLE_LAYER = {number: number // 100 for number in CONDUCTORS}
CONDUCTOR_COLOURS = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(CONDUCTORS)))

# Strand temperature (K) separating the subcooled baseline (~6 K) from a
# quenched snapshot; used to pick the last snapshot before thermal runaway.
QUENCH_TEMPERATURE_THRESHOLD = 10.0

# Left edge (event time) of the zoomed hot-spot time evolutions.
QUENCH_ZOOM_START = -2.0

# Detection-aligned event-time convention. ``has_quench`` is False for runs
# that never go normal (no V_DL crosses the detection threshold); those are
# placed on the nominal measured schedule instead of a model detection.
Alignment = namedtuple("Alignment", ("reference_time", "offset", "has_quench"))

# Component -> report label for the hot-spot figures.
SOLID_COMPONENTS = (
    ("STR_MIX1", "strand"),
    ("Z_JACKET1", "jacket"),
    ("Z_JACKET2", "epoxy"),
)


# --------------------------------------------------------------------------
# Shared helpers.
# --------------------------------------------------------------------------
def lighten(colour, amount: float = DETECTION_LIGHTEN):
    """Mix ``colour`` toward white by ``amount`` (0 = unchanged, 1 = white)."""
    red, green, blue = mcolors.to_rgb(colour)
    return tuple(amount * (1.0 - channel) for channel in (red, green, blue))


def in_window(time, value, window):
    """Restrict a (time, value) model trace to ``window`` so a tighter axis
    also tightens the autoscaled y-range."""
    time = np.asarray(time)
    value = np.asarray(value)
    mask = (time >= window[0]) & (time <= window[1])
    return time[mask], value[mask]


def to_event_time(simulation_time, alignment: Alignment):
    """Simulation time -> detection-aligned event time (quench near t = 0).

    Same convention as the overlay panels: the measured quench detection
    sits at +0.461 s, so the modelled detection is shifted to match. Runs
    that never quench are placed on the nominal measured schedule.
    """
    return simulation_time - alignment.reference_time + alignment.offset



def save_panels(panels: dict, output_directory: Path) -> list[Path]:
    """Draw each panel on its own single-axis figure and save it as EPS.

    ``panels`` maps an output stem to a callable that draws onto a fresh
    axis. The detection-aligned overlay panels use a wider figure so their
    dense legends stay readable; the others use the report default.
    """
    written = []
    for stem, draw in panels.items():
        figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        draw(axis)
        axis.grid(alpha=0.3)
        eps_file = output_directory / f"{stem}.eps"
        figure.savefig(eps_file, format="eps")
        plt.close(figure)
        written.append(eps_file)
    return written


def output_directory_of(model_directory: Path) -> Path:
    matches = [
        path
        for path in model_directory.glob("**/Output/Spatial_distribution")
        if (path.parent / "Time_evolution").is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"no Output/Spatial_distribution folder under {model_directory}"
        )
    return sorted(matches)[0].parent


def read_network_history(model_directory: Path) -> pd.DataFrame:
    matches = sorted(model_directory.glob("**/hydraulic_network_te.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"no hydraulic_network_te.tsv under {model_directory}"
        )
    return pd.read_csv(matches[0], sep="\t")


def quench_alignment(log: pd.DataFrame) -> Alignment:
    """Event-time alignment for a run, from its controller log.

    If a detection voltage crosses the 0.5 V threshold, align the modelled
    detection to the measured one (+0.461 s). Otherwise the run never went
    normal, so fall back to the nominal measured schedule (simulation start
    at SIMULATION_START_MEASURED_TIME before the scheduled quench).
    """
    voltage_columns = [c for c in log.columns if c.startswith("V_DL")]
    above = log[voltage_columns].abs().max(axis=1) >= 0.5
    if above.any():
        detection_time = float(log["time"][above.idxmax()])
        return Alignment(detection_time, MEASURED_DETECTION_TIME, True)
    return Alignment(-SIMULATION_START_MEASURED_TIME, 0.0, False)


def quench_snapshots(output_root: Path) -> tuple[float, float]:
    """Return (pre, post) snapshot times in simulation seconds.

    ``post`` is the last saved spatial snapshot (after quench when the run
    quenched, otherwise the final state). ``pre`` is the last snapshot still
    subcooled before thermal runaway; for a non-quenching run it is simply
    the snapshot before ``post``.
    """
    spatial = output_root / "Spatial_distribution"
    conductors = sorted(
        path for path in spatial.iterdir() if path.name.startswith("CONDUCTOR")
    )
    columns = pd.read_csv(
        component_file(conductors[0], "STR_MIX1", "temperature_sd.tsv"),
        sep="\t", nrows=0,
    ).columns
    times = np.array(
        [float(name.split("=")[1].split("(")[0]) for name in columns]
    )
    strand_max = np.zeros(len(columns))
    for conductor in conductors:
        frame = pd.read_csv(
            component_file(conductor, "STR_MIX1", "temperature_sd.tsv"), sep="\t"
        )
        strand_max = np.maximum(strand_max, frame.max().to_numpy())

    order = np.argsort(times)
    times, strand_max = times[order], strand_max[order]
    post_index = len(times) - 1
    subcooled = [
        index
        for index in range(post_index)
        if strand_max[index] < QUENCH_TEMPERATURE_THRESHOLD
    ]
    pre_index = subcooled[-1] if subcooled else max(post_index - 1, 0)
    return float(times[pre_index]), float(times[post_index])


def component_file(conductor_directory: Path, prefix: str, suffix: str) -> Path:
    """Locate ``<prefix>..._<suffix>`` for a component in a conductor dir."""
    matches = sorted(conductor_directory.glob(f"{prefix}*_{suffix}"))
    if not matches:
        raise FileNotFoundError(
            f"no {prefix}*_{suffix} in {conductor_directory}"
        )
    return matches[0]


def snapshot_column(frame: pd.DataFrame, target_time: float) -> str:
    """Return the column whose 'time = X (s)' header is nearest ``target``."""
    times = np.array(
        [float(name.split("=")[1].split("(")[0]) for name in frame.columns]
    )
    return frame.columns[int(np.abs(times - target_time).argmin())]


# --------------------------------------------------------------------------
# Group 1: individual EPS versions of the overlay panels.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Group 2: hot-spot temperature time evolution per conductor.
# --------------------------------------------------------------------------
def hotspot_panels(
    output_root: Path, alignment: Alignment, hotspot_log=None
) -> dict:
    panels: dict = {}
    time_evolution = output_root / "Time_evolution"
    reference = "quench at t = 0" if alignment.has_quench else "nominal schedule"

    def hotspot_trace(conductor, prefix, label):
        """True per-step spatial maximum from the controller's hotspot log
        when available; otherwise the 3-station te probes (which under-read
        the peak between stations)."""
        double_layer = DOUBLE_LAYER[conductor]
        column = f"{label}_max_DL{double_layer}"
        if hotspot_log is not None and column in hotspot_log:
            time = to_event_time(
                hotspot_log["time"].to_numpy(), alignment
            )
            return time, hotspot_log[column].to_numpy()
        frame = pd.read_csv(
            component_file(
                time_evolution / f"CONDUCTOR_{conductor}",
                prefix, "temperature_te.tsv",
            ),
            sep="\t",
        )
        time = to_event_time(frame.iloc[:, 0].to_numpy(), alignment)
        return time, frame.iloc[:, 1:].max(axis=1).to_numpy()

    def make_draw(conductor, zoom=False):
        def draw(axis):
            # Strand, jacket and epoxy nearly coincide near quench; draw the
            # strand on top (highest z-order) so it stays visible.
            for (prefix, label), colour, zorder in zip(
                SOLID_COMPONENTS, ("tab:red", "tab:blue", "tab:green"), (5, 4, 3)
            ):
                time, hotspot = hotspot_trace(conductor, prefix, label)
                if zoom:
                    # Rescale the vertical axis to the zoom window so the
                    # quench rise fills the panel.
                    visible = time >= QUENCH_ZOOM_START
                    axis.plot(time[visible], hotspot[visible], "-", lw=1.2,
                              color=colour, label=label, zorder=zorder)
                else:
                    axis.plot(time, hotspot, "-", lw=1.2, color=colour,
                              label=label, zorder=zorder)
            axis.axvline(0.0, color="gray", ls=":", lw=0.7)
            title = f"DL{DOUBLE_LAYER[conductor]} hot-spot temperature"
            axis.set_title(f"{title} (quench event)" if zoom else title)
            axis.set_xlabel(f"event time [s] ({reference})")
            axis.set_ylabel("max temperature [K]")
            if zoom:
                axis.set_xlim(left=QUENCH_ZOOM_START)
            axis.legend(fontsize=9)
        return draw

    for conductor in CONDUCTORS:
        double_layer = DOUBLE_LAYER[conductor]
        panels[f"hotspot_temperature_DL{double_layer}"] = make_draw(conductor)
        # A zoomed quench close-up only makes sense when the run quenches.
        if alignment.has_quench:
            panels[f"hotspot_temperature_DL{double_layer}_zoom"] = make_draw(
                conductor, zoom=True
            )

    def draw_all_strands(axis):
        # Strand hot-spot of every DL on one axis, zoomed on the quench event,
        # so the order in which the layers run away is visible.
        for conductor, colour in zip(CONDUCTORS, CONDUCTOR_COLOURS):
            time, hotspot = hotspot_trace(conductor, "STR_MIX1", "strand")
            visible = time >= QUENCH_ZOOM_START
            axis.plot(time[visible], hotspot[visible], "-", lw=1.2,
                      color=colour, label=f"DL{DOUBLE_LAYER[conductor]}")
        axis.axvline(0.0, color="gray", ls=":", lw=0.7)
        title = "strand hot-spot temperature"
        axis.set_title(f"{title} (quench event)"
                       if alignment.has_quench else title)
        axis.set_xlabel(f"event time [s] ({reference})")
        axis.set_ylabel("max temperature [K]")
        axis.set_xlim(left=QUENCH_ZOOM_START)
        axis.legend(fontsize=8, ncol=2)

    panels["strand_hotspot_all_dl_zoom"] = draw_all_strands
    return panels


# --------------------------------------------------------------------------
# Groups 3 & 4: spatial distributions, all conductors overlaid, per snapshot.
# --------------------------------------------------------------------------
def spatial_panels(
    output_root: Path, alignment: Alignment, pre_time: float, post_time: float
) -> dict:
    panels: dict = {}
    spatial = output_root / "Spatial_distribution"

    def load_profile(conductor, prefix, quantity, target_time):
        directory = spatial / f"CONDUCTOR_{conductor}"
        frame = pd.read_csv(
            component_file(directory, prefix, f"{quantity}_sd.tsv"), sep="\t"
        )
        zcoord = pd.read_csv(directory / "zcoord.tsv", sep="\t")
        column = snapshot_column(frame, target_time)
        z_column = snapshot_column(zcoord, target_time)
        return zcoord[z_column].to_numpy(), frame[column].to_numpy()

    def make_overlay(prefix, quantity, target_time, title, ylabel, scale=1.0):
        def draw(axis):
            for conductor, colour in zip(CONDUCTORS, CONDUCTOR_COLOURS):
                z, value = load_profile(conductor, prefix, quantity, target_time)
                axis.plot(z, value * scale, "-", lw=1.0, color=colour,
                          label=f"DL{DOUBLE_LAYER[conductor]}")
            axis.set_title(title)
            axis.set_xlabel("position along conductor [m]")
            axis.set_ylabel(ylabel)
            axis.legend(fontsize=8, ncol=2)
        return draw

    post_event = to_event_time(post_time, alignment)
    pre_event = to_event_time(pre_time, alignment)
    if alignment.has_quench:
        post_caption = f"after quench, t = {post_event:+.1f} s"
        pre_caption = f"before quench, t = {pre_event:+.1f} s"
    else:
        post_caption = f"final state, t = {post_event:+.1f} s"
        pre_caption = f"t = {pre_event:+.1f} s"
    snapshots = {
        "postquench": (post_time, post_caption),
        "prequench": (pre_time, pre_caption),
    }

    # Outputs written before the mass_flow_rate spatial column existed lack
    # the CHAN1*_mass_flow_rate_sd.tsv files; skip those panels then.
    has_mass_flow = all(
        sorted(
            (spatial / f"CONDUCTOR_{conductor}").glob(
                "CHAN1*_mass_flow_rate_sd.tsv"
            )
        )
        for conductor in CONDUCTORS
    )

    for tag, (target_time, caption) in snapshots.items():
        panels[f"strand_temperature_distribution_{tag}"] = make_overlay(
            "STR_MIX1", "temperature", target_time,
            f"Strand temperature ({caption})", "temperature [K]",
        )
        panels[f"channel_pressure_distribution_{tag}"] = make_overlay(
            "CHAN1", "pressure", target_time,
            f"Channel pressure ({caption})", "pressure [bar]", scale=1.0 / BAR,
        )
        panels[f"channel_velocity_distribution_{tag}"] = make_overlay(
            "CHAN1", "velocity", target_time,
            f"Channel velocity ({caption})", "velocity [m/s]",
        )
        if has_mass_flow:
            panels[f"channel_mass_flow_distribution_{tag}"] = make_overlay(
                "CHAN1", "mass_flow_rate", target_time,
                f"Channel mass flow rate ({caption})",
                "mass flow rate [g/s]", scale=1e3,
            )
    return panels


def cea_comparison_figure(
    model_directory: Path, alignment: Alignment, output_directory: Path
) -> Path:
    """Two-panel outlet-temperature comparison with the CEA measurements.

    Top: outlet 1 measured (TT2100A) against the 1D THEA run and this (2D)
    coupled run. Bottom: outlets 2-4, measured vs 2D. The THEA file is
    already on the measured event-time axis (thea_1d convention).
    """
    network = read_network_history(model_directory)
    model_time = to_event_time(network["time (s)"].to_numpy(), alignment)

    thea = pd.read_csv(THEA_ONE_D_FILE)
    one_d_time = thea["Time"].to_numpy()
    one_d_outlet = thea["Strand Temperature"].to_numpy()

    label_size = 10 * CEA_FONT_SCALE
    legend_size = 8 * CEA_FONT_SCALE
    title_size = 12 * CEA_FONT_SCALE

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), sharex=True, constrained_layout=True
    )

    sensor = GROUP_TEMPERATURE_SENSOR[1]
    measured_time, measured = md.merged_schedule(
        f"{sensor}_ai", sensor, window=CEA_PLOT_WINDOW, slow_tail=True
    )
    time_1d, outlet_1d = in_window(one_d_time, one_d_outlet, CEA_PLOT_WINDOW)
    top.plot(time_1d, outlet_1d, "-", color=MPG_GREEN_LIGHT, marker="D",
             markevery=0.05, markersize=6, label="Outlet 1, 1D")
    time_2d, outlet_2d = in_window(
        model_time, network["temperature_outlet1_header (K)"], CEA_PLOT_WINDOW
    )
    top.plot(time_2d, outlet_2d, "-", color=MPG_GREEN, marker="o",
             markevery=0.05, markersize=6, label="Outlet 1, 2D")
    top.plot(measured_time, measured, "--", color=MPG_BLUE_LIGHT,
             label="Outlet 1, meas.")
    for values, colour, tag in (
        (outlet_1d, MPG_GREEN_LIGHT, "1D"),
        (outlet_2d, MPG_GREEN, "2D"),
        (measured, MPG_BLUE_LIGHT, "meas"),
    ):
        peak = float(np.max(values))
        top.axhline(peak, color=colour, linestyle="-.", linewidth=1.2,
                    zorder=1)
        top.text(
            CEA_PLOT_WINDOW[0] + 0.3, peak - 1.5,
            rf"$T_{{max,\,{tag}}} = {peak:.1f}$ K",
            fontsize=legend_size, color=colour, ha="left", va="top",
        )
    peak_1d = float(np.max(outlet_1d))
    peak_2d = float(np.max(outlet_2d))
    peak_meas = float(np.max(measured))
    for upper, lower, text_shift in (
        (peak_1d, peak_2d, 0.0),
        (peak_2d, peak_meas, -3.0),
    ):
        top.annotate(
            "",
            xy=(5.0, upper),
            xytext=(5.0, lower),
            arrowprops={
                "arrowstyle": "<->",
                "color": "red",
                "linewidth": 2.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        top.text(
            5.3, 0.5 * (upper + lower) + text_shift,
            rf"$\Delta T = {upper - lower:.1f}$ K",
            fontsize=legend_size, color="red", ha="left", va="center",
        )
    top.set_ylabel("Temperature (K)", fontsize=label_size)
    top.tick_params(labelsize=label_size)
    top.legend(fontsize=legend_size, loc="upper right")
    top.grid(alpha=0.3)

    for group in (2, 3, 4):
        colour = GROUP_COLOURS[group]
        sensor = GROUP_TEMPERATURE_SENSOR[group]
        measured_time, measured = md.merged_schedule(
            f"{sensor}_ai", sensor, window=CEA_PLOT_WINDOW, slow_tail=True
        )
        model_t, model_temperature = in_window(
            model_time,
            network[f"temperature_outlet{group}_header (K)"],
            CEA_PLOT_WINDOW,
        )
        bottom.plot(model_t, model_temperature, "-", color=colour,
                    marker="o", markevery=0.05, markersize=6,
                    label=f"Outlet {group}, 2D")
        bottom.plot(measured_time, measured, "--", color=colour,
                    label=f"Outlet {group}, meas.")
    bottom.set_ylabel("Temperature (K)", fontsize=label_size)
    bottom.set_xlabel("Event time (s)", fontsize=label_size)
    bottom.set_xlim(*CEA_PLOT_WINDOW)
    bottom.tick_params(labelsize=label_size)
    bottom.legend(fontsize=legend_size, ncol=1, loc="upper left")
    bottom.grid(alpha=0.3)

    figure.suptitle(
        "Comparison with quench measurements at CEA (2006)",
        fontsize=title_size,
    )
    eps_file = output_directory / "outlet_temperatures_cea_comparison.eps"
    figure.savefig(eps_file, format="eps")
    plt.close(figure)
    return eps_file


def generate_for_model(
    model_directory: Path, output_directory: Path,
) -> None:
    """Standard figure set of one run into ``output_directory``: hot-spot
    time evolutions, spatial snapshots and the CEA outlet comparison."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_root = output_directory_of(model_directory)

    # Event-time alignment and the pre/post-quench spatial snapshots are
    # detected from the run itself, so the same script serves every variant.
    log = pd.read_csv(model_directory / "final_run_log.tsv", sep="\t")
    alignment = quench_alignment(log)
    pre_time, post_time = quench_snapshots(output_root)

    panels: dict = {}
    hotspot_file = model_directory / "hotspot_log.tsv"
    hotspot_log = (
        pd.read_csv(hotspot_file, sep="\t") if hotspot_file.exists() else None
    )
    panels.update(hotspot_panels(output_root, alignment, hotspot_log))
    panels.update(spatial_panels(output_root, alignment, pre_time, post_time))

    written = save_panels(panels, output_directory)
    written.append(
        cea_comparison_figure(model_directory, alignment, output_directory)
    )
    quench = (
        f"quench at t_sim = {alignment.reference_time:.1f} s"
        if alignment.has_quench
        else "no model quench (nominal schedule)"
    )
    print(
        f"{model_directory.name}: wrote {len(written)} EPS figures to "
        f"{output_directory} [{quench}; snapshots "
        f"{pre_time:.1f}/{post_time:.1f} s]"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, nargs="+", required=True,
        help="one or more run directories to process",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="figure directory (only valid with a single --model-dir; "
        "defaults to <model-dir>/figures for each run)",
    )
    arguments = parser.parse_args(argv)

    if arguments.output_dir is not None and len(arguments.model_dir) > 1:
        parser.error("--output-dir cannot be combined with several model dirs")

    for model_directory in arguments.model_dir:
        output_directory = arguments.output_dir or (model_directory / "figures")
        generate_for_model(model_directory, output_directory)


if __name__ == "__main__":
    main()
