"""Publication figures of this run -> publication-ready/*.eps.

Reuses the data access of generate_figures.py (same directory). Common
rules: units in round brackets, time axis "event time (s)", fonts 1.25 x
the matplotlib default, no titles.

Usage:
    ../../../.venv/bin/python publication_figures.py [--model-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import figure_library as gf  # noqa: E402

md = gf.md

FONT_SCALE = 1.25
BASE_FONT = 10.0 * FONT_SCALE          # 12.5 pt
LEGEND_FONT = 8.0 * FONT_SCALE         # 10 pt
CEA_LEGEND_FONT = LEGEND_FONT * 1.25   # 12.5 pt, legends of the CEA figure
DUMP_START = 0.645                     # s event time, current dump start

MPG_GREEN = (0 / 255, 108 / 255, 102 / 255)
MPG_PURPLE = (128 / 255, 0 / 255, 128 / 255)

# Viridis, DL1 darkest ... DL6 lightest; one marker shape per DL so the
# curves stay distinguishable without colour.
DL_VIRIDIS = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, 6))
DL_MARKERS = ("o", "s", "D", "^", "v", "P")


def event_axis(axis, window, dump_marker=True):
    if dump_marker:
        axis.axvline(DUMP_START, color="gray", ls=":", lw=0.8)
    axis.set_xlim(*window)
    axis.set_xlabel("event time (s)")
    axis.grid(alpha=0.3)


def legend_below(figure, ncol, handles=None, labels=None):
    arguments = {} if handles is None else {"handles": handles, "labels": labels}
    figure.legend(loc="outside lower center", ncol=ncol,
                  fontsize=LEGEND_FONT, frameon=True, **arguments)


def save(figure, output_directory: Path, stem: str) -> Path:
    target = output_directory / f"{stem}.eps"
    figure.savefig(target, format="eps")
    plt.close(figure)
    print(f"[figure] {target}")
    return target


# 1. Detection voltage --------------------------------------------------
def least_squares_spline(time, value, coarse=0.2, fine=0.004,
                         fine_window=(0.25, 0.75), degree=3, minimum=4):
    """Least-squares cubic B-spline through a noisy record.

    Interior knots every ``coarse`` seconds, densified to ``fine`` inside
    ``fine_window`` (the quench rise, clip and dump step of the detection
    voltage). Knots whose interval would hold fewer than ``minimum``
    samples are dropped so the Schoenberg-Whitney condition holds on
    irregularly sampled data. Returns (fit time, fit value) on a dense
    grid.
    """
    from scipy.interpolate import make_lsq_spline

    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    a, b = fine_window
    candidates = np.concatenate((
        np.arange(time[0] + coarse, a, coarse),
        np.arange(a, b, fine),
        np.arange(b, time[-1] - coarse / 2, coarse),
    ))
    knots, last = [], time[0]
    for candidate in candidates:
        if np.count_nonzero((time >= last) & (time < candidate)) >= minimum:
            knots.append(candidate)
            last = candidate
    if knots and np.count_nonzero(time >= knots[-1]) < minimum:
        knots.pop()
    knot_vector = np.concatenate((
        [time[0]] * (degree + 1), knots, [time[-1]] * (degree + 1)
    ))
    spline = make_lsq_spline(time, value, knot_vector, k=degree)
    dense = np.linspace(time[0], time[-1], 20_000)
    return dense, spline(dense)


def smoothed_measurement(measured_time, measured, fast_spacing=1e-3):
    """Fit the fast (event) record with the least-squares spline and keep
    the sparse slow-archive samples before it as they are, joined to the
    start of the fit."""
    measured_time = np.asarray(measured_time, dtype=float)
    measured = np.asarray(measured, dtype=float)
    spacing = np.diff(measured_time)
    fast = np.flatnonzero(spacing < fast_spacing)
    start = int(fast[0]) if fast.size else 0
    fit_time, fit = least_squares_spline(measured_time[start:], measured[start:])
    return (np.concatenate((measured_time[:start], fit_time)),
            np.concatenate((measured[:start], fit)))


DETECTION_WINDOW = (-3.0, 3.0)


def draw_detection_voltage(axis, log, log_time, smoothed=False,
                           window=DETECTION_WINDOW):
    """Measured DUP11-3 trace (raw, or its least-squares spline with
    ``smoothed`` so the pre-quench noise band does not dominate the log
    axis) against the modelled double-layer voltages, into ``axis``.
    Returns (handles, labels) in legend order."""
    measured_time, measured = md.merged_schedule(
        f"{gf.DETECTION_SENSOR}_ai", gf.DETECTION_SENSOR, window=window
    )
    if smoothed:
        measured_time, measured = smoothed_measurement(measured_time, measured)
        label, style = "DUP11-3, least-squares spline (clips 1V)", "r--"
    else:
        label, style = "DUP11-3 (clips 1V)", "k-"
    # The smoothed measurement is drawn on top of the modelled curves.
    axis.plot(measured_time, np.abs(measured) + 1e-6, style, lw=1.8,
              label=label, zorder=10 if smoothed else 2)
    for double_layer, marker in zip(range(1, 7), DL_MARKERS):
        axis.plot(log_time, np.abs(log[f"V_DL{double_layer}"]) + 1e-9, "-",
                  color=DL_VIRIDIS[double_layer - 1], lw=1.8,
                  marker=marker, markevery=0.05, markersize=6,
                  label=f"DL{double_layer}")
    axis.set_yscale("log")
    axis.set_ylabel("detection voltage (V)")
    event_axis(axis, window)
    handles, labels = axis.get_legend_handles_labels()
    if smoothed:
        # Measurement as the last legend entry, after DL1..DL6.
        handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    return handles, labels


def detection_voltage(log, log_time, output_directory, smoothed=False):
    figure, axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    handles, labels = draw_detection_voltage(axis, log, log_time, smoothed)
    legend_below(figure, ncol=4, handles=handles, labels=labels)
    stem = "detection_voltage_smoothed" if smoothed else "detection_voltage"
    return save(figure, output_directory, stem)


# 2. Outlet temperatures of all groups ----------------------------------
def outlet_temperatures(network, model_time, output_directory,
                        adiabatic: bool = False):
    """All four outlet groups, simulated (thick) vs measured (dashed).
    With ``adiabatic`` the 1D THEA outlet-1 trace is added as a thick
    purple curve (taller figure, three legend rows)."""
    window = (0.0, 9.0)
    height = 5.4 if adiabatic else 4.2
    figure, axis = plt.subplots(figsize=(8.0, height), constrained_layout=True)
    if adiabatic:
        thea = pd.read_csv(gf.THEA_ONE_D_FILE)
        time_1d, outlet_1d = gf.in_window(
            thea["Time"].to_numpy(), thea["Strand Temperature"].to_numpy(),
            window,
        )
        axis.plot(time_1d, outlet_1d, "-", color=MPG_PURPLE, lw=2.5,
                  label="Outlet 1 simulated (adiabatic)")
    for group in (1, 2, 3, 4):
        colour = gf.GROUP_COLOURS[group]
        sensor = gf.GROUP_TEMPERATURE_SENSOR[group]
        measured_time, measured = md.merged_schedule(
            f"{sensor}_ai", sensor, window=window, slow_tail=True
        )
        header = network[f"temperature_outlet{group}_header (K)"]
        model_t, model_header = gf.in_window(model_time, header, window)
        axis.plot(model_t, model_header, "-", color=colour, lw=2.5,
                  label=f"Outlet {group} simulated")
        axis.plot(measured_time, measured, "--", color=colour, lw=1.5,
                  label=f"Outlet {group} measured")
    axis.set_ylabel("outlet temperature (K)")
    event_axis(axis, window)
    legend_below(figure, ncol=3 if adiabatic else 4)
    stem = "outlet_temperatures_all_groups"
    if adiabatic:
        stem += "_adiabatic"
    return save(figure, output_directory, stem)


# 3. CEA comparison (WP vs adiabatic vs measured) -----------------------
def cea_comparison(network, model_time, output_directory):
    window = gf.CEA_PLOT_WINDOW
    thea = pd.read_csv(gf.THEA_ONE_D_FILE)
    time_1d, outlet_1d = gf.in_window(
        thea["Time"].to_numpy(), thea["Strand Temperature"].to_numpy(), window
    )
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True, constrained_layout=True
    )
    draw_cea_outlet_1(top, network, model_time, time_1d, outlet_1d, window)
    draw_cea_groups_234(bottom, network, model_time, window)
    return save(figure, output_directory, "outlet_temperatures_cea_comparison")


def draw_cea_outlet_1(top, network, model_time, time_1d, outlet_1d, window):
    """Outlet 1: adiabatic (THEA), WP model and measurement with the peak
    levels and the two peak differences (the upper CEA panel)."""
    sensor = gf.GROUP_TEMPERATURE_SENSOR[1]
    measured_time, measured = md.merged_schedule(
        f"{sensor}_ai", sensor, window=window, slow_tail=True
    )
    top.plot(time_1d, outlet_1d, "-", color=MPG_PURPLE, marker="D",
             markevery=0.05, markersize=6, label="Outlet 1, adiabatic")
    time_2d, outlet_2d = gf.in_window(
        model_time, network["temperature_outlet1_header (K)"], window
    )
    top.plot(time_2d, outlet_2d, "-", color=gf.MPG_GREEN, marker="o",
             markevery=0.05, markersize=6, label="Outlet 1, WP")
    top.plot(measured_time, measured, "--", color="black",
             label="Outlet 1, meas.")
    for values, colour, tag in (
        (outlet_1d, MPG_PURPLE, "adiabatic"),
        (outlet_2d, gf.MPG_GREEN, "WP"),
        (measured, "black", "meas"),
    ):
        peak = float(np.max(values))
        top.axhline(peak, color=colour, linestyle="-.", linewidth=1.2, zorder=1)
        top.text(
            window[0] + 0.3, peak - 1.5,
            rf"$T_{{max,\,{tag}}} = {peak:.1f}$ K",
            fontsize=LEGEND_FONT, color=colour, ha="left", va="top",
        )
    peak_1d, peak_2d, peak_meas = (
        float(np.max(outlet_1d)), float(np.max(outlet_2d)), float(np.max(measured))
    )
    for upper, lower, text_shift in (
        (peak_1d, peak_2d, 0.0),
        (peak_2d, peak_meas, -3.0),
    ):
        top.annotate(
            "", xy=(5.0, upper), xytext=(5.0, lower),
            arrowprops={"arrowstyle": "<->", "color": "red", "linewidth": 2.0,
                        "shrinkA": 0.0, "shrinkB": 0.0},
        )
        top.text(
            5.3, 0.5 * (upper + lower) + text_shift,
            rf"$\Delta T = {upper - lower:.1f}$ K",
            fontsize=LEGEND_FONT, color="red", ha="left", va="center",
        )
    top.set_ylabel("temperature (K)")
    top.legend(fontsize=CEA_LEGEND_FONT, loc="upper right")
    top.grid(alpha=0.3)


def cea_comparison_outlet_1(network, model_time, output_directory,
                            width=8.0, suffix=""):
    """The upper CEA panel (outlet 1) as a figure of its own."""
    window = gf.CEA_PLOT_WINDOW
    thea = pd.read_csv(gf.THEA_ONE_D_FILE)
    time_1d, outlet_1d = gf.in_window(
        thea["Time"].to_numpy(), thea["Strand Temperature"].to_numpy(), window
    )
    figure, axis = plt.subplots(figsize=(width, 4.0), constrained_layout=True)
    draw_cea_outlet_1(axis, network, model_time, time_1d, outlet_1d, window)
    axis.set_xlabel("event time (s)")
    axis.set_xlim(*window)
    return save(figure, output_directory,
                f"outlet_temperatures_cea_outlet_1{suffix}")


def draw_cea_groups_234(axis, network, model_time, window, legend=True):
    """Outlets 2-4, WP model against the measured sensors (the lower panel
    of the CEA comparison). With ``network`` None (a run without the
    hydraulic network) only the measured sensors are drawn."""
    for group in (2, 3, 4):
        colour = gf.GROUP_COLOURS[group]
        sensor = gf.GROUP_TEMPERATURE_SENSOR[group]
        measured_time, measured = md.merged_schedule(
            f"{sensor}_ai", sensor, window=window, slow_tail=True
        )
        if network is not None:
            model_t, model_temperature = gf.in_window(
                model_time, network[f"temperature_outlet{group}_header (K)"],
                window,
            )
            axis.plot(model_t, model_temperature, "-", color=colour,
                      marker="o", markevery=0.05, markersize=6,
                      label=f"Outlet {group}, WP")
        axis.plot(measured_time, measured, "--", color=colour,
                  label=f"Outlet {group}, meas.")
    axis.set_ylabel("temperature (K)")
    axis.set_xlabel("event time (s)")
    axis.set_xlim(*window)
    if legend:
        axis.legend(fontsize=CEA_LEGEND_FONT, ncol=1, loc="upper left")
    axis.grid(alpha=0.3)


def cea_comparison_groups_234(network, model_time, output_directory,
                              width=8.0, suffix=""):
    """The lower CEA panel (outlets 2-4) as a figure of its own; ``width``
    in inches (the ``_narrow`` variant uses 6.4)."""
    figure, axis = plt.subplots(figsize=(width, 4.0), constrained_layout=True)
    draw_cea_groups_234(axis, network, model_time, gf.CEA_PLOT_WINDOW)
    return save(figure, output_directory,
                f"outlet_temperatures_cea_groups_2-4{suffix}")


# 4. Coil inlet: TT2104A vs filter node ---------------------------------
def outlet_flows(network, model_time, output_directory):
    window = (-1.0, 9.0)
    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for group in (1, 2, 3, 4):
        flow = network[f"mass_flow_rate_outlet{group}_to_return (kg/s)"] * 1e3
        model_t, model_flow = gf.in_window(model_time, flow, window)
        axis.plot(model_t, model_flow, "-", color=gf.GROUP_COLOURS[group],
                  lw=2.0, label=f"Outlet {group}")
    axis.axhline(0.0, color="black", ls="--", lw=2.0, zorder=1)
    axis.set_ylabel("mass flow rate (g/s)")
    axis.legend(fontsize=LEGEND_FONT)
    event_axis(axis, window)
    return save(figure, output_directory, "outlet_flows")


# 8. Computed mass flow at the double-layer inlets ---------------------
DL_FLOW_WINDOW = (-1.0, 9.0)


def draw_dl_end_flows(axis, model_directory, alignment, end, window=DL_FLOW_WINDOW):
    """Draw the channel mass flow at the ``end`` ("inlet" or "outlet") of
    every double layer (positive = in the nominal flow direction, inlet to
    outlet), from the CHAN1 inlet/outlet time-evolution files."""
    column = {"inlet": "mass_flow_rate_inl (kg/s)",
              "outlet": "mass_flow_rate_out (kg/s)"}[end]
    time_evolution = gf.output_directory_of(model_directory) / "Time_evolution"
    for conductor, marker in zip(gf.CONDUCTORS, DL_MARKERS):
        frame = pd.read_csv(
            gf.component_file(
                time_evolution / f"CONDUCTOR_{conductor}", "CHAN1",
                "inlet_outlet_te.tsv",
            ),
            sep="\t",
        )
        time = gf.to_event_time(frame["time (s)"].to_numpy(), alignment)
        flow = frame[column].to_numpy() * 1e3
        model_t, model_flow = gf.in_window(time, flow, window)
        double_layer = gf.DOUBLE_LAYER[conductor]
        axis.plot(model_t, model_flow, "-", lw=2.0,
                  color=DL_VIRIDIS[double_layer - 1], marker=marker,
                  markevery=0.05, markersize=6, label=f"DL{double_layer}")
    axis.axhline(0.0, color="black", ls="--", lw=2.0, zorder=1)
    axis.set_ylabel(f"{end} mass flow rate (g/s)")
    event_axis(axis, window)


def dl_end_flows(model_directory, alignment, output_directory, end="inlet"):
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    draw_dl_end_flows(axis, model_directory, alignment, end)
    legend_below(figure, ncol=6)
    return save(figure, output_directory, f"dl_{end}_flows")


def dl_flows_combined(model_directory, alignment, output_directory):
    """Inlet (left) and outlet (right) DL flows side by side, one shared
    legend below."""
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), constrained_layout=True
    )
    draw_dl_end_flows(left, model_directory, alignment, "inlet")
    draw_dl_end_flows(right, model_directory, alignment, "outlet")
    left.set_title("Inlet flows")
    right.set_title("Outlet flows")
    left.set_ylabel("mass flow rate (g/s)")
    right.set_ylabel("mass flow rate (g/s)")
    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6,
                  fontsize=LEGEND_FONT, frameon=True)
    return save(figure, output_directory, "dl_flows_inlet_outlet")


# 7. Strand temperature profiles at a saved snapshot -------------------
def spatial_profiles(model_directory, alignment, event_time, output_directory,
                     prefix, quantity, ylabel, stem_name, scale=1.0,
                     width=8.0, suffix="", double_layers=None):
    """One quantity along every DL channel at the saved spatial snapshot
    nearest to ``event_time`` (event seconds); ``width`` in inches (the
    ``_narrow`` variant uses 6.4). ``double_layers`` restricts the figure
    to a subset (each layer keeps its colour and marker from the all-DL
    figure, and the stem names the selection)."""
    selection = tuple(double_layers) if double_layers else None
    figure, axis = plt.subplots(figsize=(width, 4.6), constrained_layout=True)
    snapshot_event = draw_spatial_profiles(
        axis, model_directory, alignment, event_time, prefix, quantity,
        ylabel, scale, selection,
    )
    legend_below(figure, ncol=len(selection) if selection else 6)
    name = stem_name
    if selection:
        name += "_" + "".join(f"DL{n}" for n in selection)
    stem = f"{name}_{snapshot_event:+.1f}s".replace("+", "") + suffix
    print(f"[snapshot] {stem_name}: requested {event_time:+.1f} s, "
          f"saved {snapshot_event:+.2f} s")
    return save(figure, output_directory, stem)


def draw_spatial_profiles(axis, model_directory, alignment, event_time,
                          prefix, quantity, ylabel, scale=1.0, selection=None):
    """Draw one quantity along every DL channel (or ``selection``) at the
    saved snapshot nearest to ``event_time`` into ``axis``; returns the
    event time of the snapshot actually used."""
    output_root = gf.output_directory_of(model_directory)
    spatial = output_root / "Spatial_distribution"
    target_time = event_time - alignment.offset + alignment.reference_time
    actual = None
    for conductor, marker in zip(gf.CONDUCTORS, DL_MARKERS):
        if selection and gf.DOUBLE_LAYER[conductor] not in selection:
            continue
        directory = spatial / f"CONDUCTOR_{conductor}"
        frame = pd.read_csv(
            gf.component_file(directory, prefix, f"{quantity}_sd.tsv"),
            sep="\t",
        )
        zcoord = pd.read_csv(directory / "zcoord.tsv", sep="\t")
        column = gf.snapshot_column(frame, target_time)
        actual = float(column.split("=")[1].split("(")[0])
        double_layer = gf.DOUBLE_LAYER[conductor]
        axis.plot(
            zcoord[gf.snapshot_column(zcoord, target_time)].to_numpy(),
            frame[column].to_numpy() * scale, "-", lw=1.5,
            color=DL_VIRIDIS[double_layer - 1], marker=marker,
            markevery=0.05, markersize=6, label=f"DL{double_layer}",
        )
    axis.set_xlabel("x (m)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    return gf.to_event_time(actual, alignment)


def strand_profiles(model_directory, alignment, event_time, output_directory,
                    width=8.0, suffix=""):
    """Strand temperature along every DL channel."""
    return spatial_profiles(
        model_directory, alignment, event_time, output_directory,
        "STR_MIX1", "temperature", "strand temperature (K)",
        "strand_temperature_profiles", 1.0, width, suffix,
    )


def channel_pressure_profiles(model_directory, alignment, event_time,
                              output_directory, width=8.0, suffix="",
                              double_layers=None):
    """Channel pressure along every DL channel (or a subset)."""
    return spatial_profiles(
        model_directory, alignment, event_time, output_directory,
        "CHAN1", "pressure", "pressure (bar)",
        "channel_pressure_profiles", 1.0 / gf.BAR, width, suffix,
        double_layers,
    )


# 5. Strand hot spot of every DL ----------------------------------------
def draw_strand_hotspots(axis, hotspot_log, alignment):
    time = gf.to_event_time(hotspot_log["time"].to_numpy(), alignment)
    visible = time >= gf.QUENCH_ZOOM_START
    for double_layer, marker in zip(range(1, 7), DL_MARKERS):
        hotspot = hotspot_log[f"strand_max_DL{double_layer}"].to_numpy()
        axis.plot(time[visible], hotspot[visible], "-", lw=1.5,
                  color=DL_VIRIDIS[double_layer - 1], marker=marker,
                  markevery=0.05, markersize=6, label=f"DL{double_layer}")
    axis.axvline(0.0, color="gray", ls=":", lw=0.8)
    axis.set_ylabel("hot-spot temperature (K)")
    axis.set_xlabel("event time (s)")
    axis.set_xlim(left=gf.QUENCH_ZOOM_START)
    axis.grid(alpha=0.3)


def strand_hotspots(hotspot_log, alignment, output_directory):
    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    draw_strand_hotspots(axis, hotspot_log, alignment)
    legend_below(figure, ncol=6)
    return save(figure, output_directory, "strand_hotspot_all_dl")


def generate(model_directory: Path, output_directory: Path | None = None) -> Path:
    """Publication figure set of one run into ``output_directory``
    (default ``<model>/publication-ready``)."""
    model_directory = Path(model_directory)
    output_directory = output_directory or model_directory / "publication-ready"
    output_directory.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.size"] = BASE_FONT
    log = pd.read_csv(model_directory / "final_run_log.tsv", sep="\t")
    alignment = gf.quench_alignment(log)
    network = gf.read_network_history(model_directory)
    model_time = gf.to_event_time(network["time (s)"].to_numpy(), alignment)
    log_time = gf.to_event_time(log["time"].to_numpy(), alignment)
    hotspot_log = pd.read_csv(model_directory / "hotspot_log.tsv", sep="\t")

    detection_voltage(log, log_time, output_directory)
    detection_voltage(log, log_time, output_directory, smoothed=True)
    outlet_temperatures(network, model_time, output_directory)
    outlet_temperatures(network, model_time, output_directory, adiabatic=True)
    cea_comparison(network, model_time, output_directory)
    cea_comparison_outlet_1(network, model_time, output_directory)
    cea_comparison_outlet_1(network, model_time, output_directory,
                            width=6.4, suffix="_narrow")
    cea_comparison_groups_234(network, model_time, output_directory)
    cea_comparison_groups_234(network, model_time, output_directory,
                              width=6.4, suffix="_narrow")
    outlet_flows(network, model_time, output_directory)
    dl_end_flows(model_directory, alignment, output_directory, end="inlet")
    dl_end_flows(model_directory, alignment, output_directory, end="outlet")
    dl_flows_combined(model_directory, alignment, output_directory)
    strand_profiles(model_directory, alignment, 2.0, output_directory)
    strand_profiles(model_directory, alignment, 2.0, output_directory,
                    width=6.4, suffix="_narrow")
    channel_pressure_profiles(model_directory, alignment, 2.0, output_directory)
    channel_pressure_profiles(model_directory, alignment, 2.0, output_directory,
                              width=6.4, suffix="_narrow")
    channel_pressure_profiles(model_directory, alignment, 2.0, output_directory,
                              double_layers=(4,))
    channel_pressure_profiles(model_directory, alignment, 2.0, output_directory,
                              width=6.4, suffix="_narrow", double_layers=(4,))
    strand_hotspots(hotspot_log, alignment, output_directory)
    return output_directory


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    generate(arguments.model_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
