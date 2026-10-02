"""Figures of the formulation verification suite, regenerated from results/
alone (state_snapshots.npz, port_timeseries.csv, diagnostics.csv,
run_manifest.json, metrics.json, tier2/reference/transmission_line.npz).

    python scripts/make_plots.py [--results-dir results] [--figures-dir figures]

Writes figures/<name>.pdf, .png (300 dpi) and <name>.txt (caption).
Style: velocity = blue, mass-flow = red; single-pass = dashed; direct
quantity = filled marker, indirect = open marker; reference = black; order
guides thin grey.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

COLOUR = {"velocity": "tab:blue", "mass_flow": "tab:red"}
LABEL = {"velocity": "velocity formulation", "mass_flow": "mass-flow formulation"}
SHORT_LABEL = {"velocity": "$v$ formulation", "mass_flow": r"$\dot{m}$ formulation"}
LINESTYLE = {"single": "--"}
# Formulation markers differ in shape (circle / diamond) so that the curves
# remain distinguishable without colour.
MARKER = {"velocity": "o", "mass_flow": "D"}
CHANNELS = ("CHAN1_C1", "CHAN1_C2", "CHAN1_C3")
LEVEL_MARKER = {"h": "o", "h2": "s", "h4": "^", "h8": "v"}


def formulation_marker(key: str) -> str:
    """Marker of a metrics key such as 'velocity', 'mass_flow_htc1', 'velocity_dx'."""
    return MARKER["velocity"] if key.startswith("velocity") else MARKER["mass_flow"]


def key_style(key: str):
    """(colour, linestyle, label) of a metrics key such as 'velocity' or
    'mass_flow_htc1' (variant suffix -> dotted line, label suffix)."""
    base, _, variant = key.partition("_htc")
    if key.endswith("_dx") or key.endswith("_dt"):
        base = key.rsplit("_", 1)[0]
    if base not in COLOUR:
        base = "velocity" if key.startswith("velocity") else "mass_flow"
    linestyle = ":" if variant else "--"
    label = LABEL[base] + (f" (floored HTC, model {variant})" if variant else "")
    return COLOUR[base], linestyle, label


def legend_handles(direct_indirect: bool = False):
    handles = [
        Line2D([], [], color=COLOUR["velocity"], linestyle="--", marker=MARKER["velocity"],
               label="velocity formulation (single-pass)"),
        Line2D([], [], color=COLOUR["mass_flow"], linestyle="--", marker=MARKER["mass_flow"],
               label="mass-flow formulation (single-pass)"),
        Line2D([], [], color="black", linestyle="-", label="reference"),
    ]
    if direct_indirect:
        handles += [
            Line2D([], [], color="grey", marker="o", linestyle="", label="direct quantity"),
            Line2D([], [], color="grey", marker="o", markerfacecolor="none", linestyle="", label="indirect quantity"),
        ]
    return handles


def order_guides(axis, x, y_anchor, orders=(1, 2), label_prefix="order"):
    x = np.asarray(x, float)
    x0, x1 = x.min(), x.max()
    for order in orders:
        y = y_anchor * (np.array([x0, x1]) / x1) ** order
        axis.plot([x0, x1], y, color="grey", linewidth=0.8, linestyle="-")
        axis.annotate(f"{label_prefix} {order}", (x0, y[0]), fontsize=7, color="grey")


def convergence_triangle(axis, x_right, y_bottom, order=1, width_factor=2.0,
                         label=r"$\mathcal{O}(\Delta x)$"):
    """Right triangle on log-log axes whose hypotenuse has the given slope:
    horizontal leg from x_right/width_factor to x_right at y_bottom, vertical
    leg at x_right up to y_bottom * width_factor**order, label at the
    logarithmic centroid."""
    x_left = x_right / width_factor
    y_top = y_bottom * width_factor ** order
    axis.plot([x_left, x_right, x_right, x_left], [y_bottom, y_bottom, y_top, y_bottom],
              color="black", linewidth=0.8, linestyle="-")
    x_text = np.exp((np.log(x_left) + 2 * np.log(x_right)) / 3)
    y_text = np.exp((2 * np.log(y_bottom) + np.log(y_top)) / 3)
    axis.text(x_text, y_text, label, fontsize=8, ha="center", va="center")


class Figures:
    def __init__(self, results: Path, figures: Path):
        self.results = results
        self.figures = figures
        figures.mkdir(parents=True, exist_ok=True)
        self.metrics = {tier: self._load_json(results / tier / "metrics.json")
                        for tier in ("tier1", "tier2", "tier3a", "tier3b", "tier4", "tier5")}
        self.point = self._load_json(results / "operating_point.json")

    @staticmethod
    def _load_json(path):
        return json.loads(path.read_text()) if path.exists() else None

    def runs(self, tier):
        root = self.results / tier
        if not root.exists():
            return {}
        out = {}
        for directory in sorted(root.iterdir()):
            if (directory / "run_manifest.json").exists():
                out[directory.name] = directory
        return out

    def save(self, figure, name, caption):
        figure.savefig(self.figures / f"{name}.pdf")
        figure.savefig(self.figures / f"{name}.png", dpi=300)
        (self.figures / f"{name}.txt").write_text(caption.strip() + "\n")
        plt.close(figure)
        print(f"[figure] {name}")

    # ------------------------------------------------------------------ F1
    def f1_tier1_flow_split(self):
        m = self.metrics["tier1"]
        if not m:
            return
        runs = m["runs"]
        levels = ["h", "h2", "h4"]
        figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
        present = [(f, l) for f in ("velocity", "mass_flow") for l in levels
                   if f"{f}_single_{l}" in runs]
        slots = 2 * len(present)
        width = 0.8 / max(slots, 1)
        positions = np.arange(len(CHANNELS))
        bottom = 1e-16
        for i, (formulation, level) in enumerate(present):
            name = f"{formulation}_single_{level}"
            eps = [runs[name]["channels"][c]["eps_mdot_mean"] for c in CHANNELS]
            eps_v = [runs[name]["channels"][c]["eps_v_max"] for c in CHANNELS]
            direct = eps if formulation == "mass_flow" else eps_v
            indirect = eps_v if formulation == "mass_flow" else eps
            x = positions - 0.4 + (2 * i + 0.5) * width
            axis.bar(x, direct, width, bottom=bottom, color=COLOUR[formulation],
                     alpha=0.4 + 0.2 * levels.index(level),
                     label=f"{LABEL[formulation]} {level}, direct")
            axis.bar(x + width, indirect, width, bottom=bottom, facecolor="none",
                     edgecolor=COLOUR[formulation], linewidth=1.2,
                     label=f"{LABEL[formulation]} {level}, indirect")
        axis.set_yscale("log")
        axis.set_ylim(bottom=1e-9)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"channel {k + 1}" for k in range(3)])
        axis.set_ylabel(r"relative error $\epsilon$ [-]")
        axis.set_title("Tier 1: steady flow split vs algebraic reference")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=6, ncol=2)
        self.save(figure, "F1_tier1_flow_split",
                  "Tier 1. Per-channel relative error of the steady mass flow (filled: direct quantity "
                  "of the formulation, i.e. mdot in the mass-flow and v in the velocity formulation; open: "
                  "the indirect one) against the algebraic steady-network reference, for both formulations "
                  "and the three refinement levels. Expected: flat at the solver tolerance / discretisation "
                  "error of the compressible profile, decreasing with refinement. A level-independent floor "
                  "indicates a friction-block bias.")

    # ------------------------------------------------------------------ F2, F3
    def _tier2_reference(self):
        path = self.results / "tier2" / "reference" / "transmission_line.npz"
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _baseline_profiles(self):
        return self.point["profiles"] if self.point else None

    def f2_tier2_pressure_profiles(self):
        runs = self.runs("tier2")
        ref = self._tier2_reference()
        if not runs or ref is None:
            return
        coarse = {f: self.results / "tier2" / f"{f}_single_h" for f in ("velocity", "mass_flow")}
        coarse = {f: d for f, d in coarse.items() if d.exists()}
        if not coarse:
            return
        manifest = json.loads(next(iter(coarse.values())).joinpath("run_manifest.json").read_text())
        transit = manifest["transit_time_channel_1"]
        step = manifest["step_amplitude"]
        baseline = self._baseline_profiles()
        figure, axes = plt.subplots(3, 4, figsize=(11, 7), constrained_layout=True, sharex="row")
        for column, factor in enumerate((0.5, 1.0, 2.0, 4.0)):
            t = factor * transit
            j = int(np.argmin(np.abs(ref["snapshot_times"] - t)))
            for row, channel in enumerate(CHANNELS):
                axis = axes[row, column]
                axis.plot(ref[f"x_{channel}"], ref[f"p_{channel}"][j] / step, color="black", linewidth=1.2)
                for formulation, directory in coarse.items():
                    data = np.load(directory / "state_snapshots.npz")
                    times = data[f"t_snap_{channel}"]
                    i = int(np.argmin(np.abs(times - t)))
                    x = data[f"x_{channel}"]
                    p0 = data[f"p_{channel}"][0]  # the run's settled state (t_rel = 0)
                    axis.plot(x, (data[f"p_{channel}"][i] - p0) / step, color=COLOUR[formulation],
                              linestyle="--", linewidth=1.0)
                axis.grid(alpha=0.3)
                if row == 0:
                    axis.set_title(f"t = {factor:g} L1/c")
                if column == 0:
                    axis.set_ylabel(f"channel {row + 1}\n" + r"$\delta p / \delta p_{step}$ [-]")
                if row == 2:
                    axis.set_xlabel("x [m]")
        figure.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=8,
                      bbox_to_anchor=(0.5, -0.02))
        figure.suptitle("Tier 2: pressure perturbation profiles, coarsest level")
        self.save(figure, "F2_tier2_pressure_profiles",
                  "Tier 2. Pressure perturbation along the three channels at 0.5, 1, 2 and 4 L1/c after "
                  "the 10 mbar supply step, transmission-line reference in black, both formulations at the "
                  "coarsest level. Expected: agreement away from the fronts; dispersion error concentrates at "
                  "the fronts and at the ports. Large interior deviations at late times indicate a wrong "
                  "wave speed or damping (property or friction linearisation), port deviations a coupling error.")

    def f3_tier2_return_port_flow(self):
        runs = self.runs("tier2")
        ref = self._tier2_reference()
        if not runs or ref is None:
            return
        channel = CHANNELS[2]
        figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        inset = axis.inset_axes([0.55, 0.12, 0.42, 0.4])
        axis.plot(ref["times"], ref[f"port_out_{channel}"], color="black", linewidth=1.2)
        inset.plot(ref["times"], ref[f"port_out_{channel}"], color="black", linewidth=1.2)
        transit = None
        for name, directory in runs.items():
            manifest = json.loads((directory / "run_manifest.json").read_text())
            transit = manifest["transit_time_channel_1"]
            ports = pd.read_csv(directory / "port_timeseries.csv")
            tier = ports[ports["time_relative"] >= 0.0]
            flow = tier[f"mdot_port_outlet_{channel}"].to_numpy()
            flow = flow - flow[0]
            level = manifest["level"]
            for target in (axis, inset):
                target.plot(tier["time_relative"], flow, color=COLOUR[manifest["formulation"]],
                            linestyle="--", linewidth=0.8 + 0.3 * ["h", "h2", "h4"].index(level) if level in ("h", "h2", "h4") else 0.8,
                            alpha=0.9)
        if transit:
            t_first = 40.0 / 30.0 * transit * 1.5  # L3/c... approximate window of the first reflection
            L3_over_c = transit * 2.0
            inset.set_xlim(L3_over_c * 0.9, L3_over_c * 1.6)
            window = (ref["times"] > L3_over_c * 0.9) & (ref["times"] < L3_over_c * 1.6)
            if window.any():
                y = ref[f"port_out_{channel}"][window]
                inset.set_ylim(y.min() - 0.1 * np.ptp(y), y.max() + 0.1 * np.ptp(y))
        inset.set_title("first return-port excursion (front arrival + reflection)", fontsize=7)
        inset.tick_params(labelsize=6)
        axis.set_xlabel("t [s]")
        axis.set_ylabel(r"$\delta \dot m$ at return port of channel 3 [kg/s]")
        axis.set_title("Tier 2: return-port mass flow of channel 3")
        axis.grid(alpha=0.3)
        axis.legend(handles=legend_handles(), fontsize=7, loc="upper left")
        self.save(figure, "F3_tier2_return_port_flow",
                  "Tier 2. Mass-flow perturbation at the return port of channel 3 (as passed to the "
                  "network) for every configuration and level, transmission-line reference in black, inset "
                  "zoom on the first reflected pulse. Expected: convergence towards the reference with "
                  "refinement; a persistent offset of the mass-flow level in the velocity formulation "
                  "indicates the old-density port-flow evaluation.")

    # ------------------------------------------------------------------ F4
    def f4_tier2_convergence(self):
        m = self.metrics["tier2"]
        if not m or not m.get("convergence"):
            return
        figure, axes = plt.subplots(2, 2, figsize=(8, 6.5), constrained_layout=True)
        panels = {("dx", "p"): axes[0, 0], ("dx", "mdot"): axes[0, 1],
                  ("dt", "p"): axes[1, 0], ("dt", "mdot"): axes[1, 1]}
        for formulation, conv in m["convergence"].items():
            for (series, variable), axis in panels.items():
                c = conv.get(series) or conv.get("prop")
                if c is None:
                    continue
                steps = np.array(c["step_sizes"])
                self_err = np.array(c[f"self_error_{variable}"])
                ref_err = np.array(c[f"reference_error_{variable}"])
                axis.loglog(steps[:-1], self_err[:-1], color=COLOUR[formulation], linestyle="--",
                            marker=MARKER[formulation],
                            label=f"{LABEL[formulation]}: vs finest ({c['levels'][-1]})")
                axis.loglog(steps, ref_err, color=COLOUR[formulation], linestyle=":",
                            marker=MARKER[formulation], markerfacecolor="none",
                            label=f"{LABEL[formulation]}: vs reference")
                if len(steps) > 1 and np.all(ref_err > 0) and not getattr(axis, "_guides", False):
                    order_guides(axis, steps, ref_err.max(), orders=(1, 2))
                    axis._guides = True
        for (series, variable), axis in panels.items():
            axis.set_xlabel(r"$\Delta x$ (m)" if series == "dx" else r"$\Delta t$ [s]")
            variable_label = "p" if variable == "p" else r"$\dot m$"
            axis.set_ylabel(f"L2 error of {variable_label} (-)")
            axis.set_title(f"{'Δx series (Δt fixed)' if series == 'dx' else 'Δt series (Δx fixed)'}: {variable}")
            axis.grid(alpha=0.3, which="both")
            axis.legend(fontsize=6)
        figure.suptitle("Tier 2: convergence at t = 4 L1/c")
        self.save(figure, "F4_tier2_convergence",
                  "Tier 2. L2 error of pressure and mass flow at 4 L1/c versus the mesh size (time step "
                  "fixed at the finest value) and versus the time step (mesh fixed at the finest), both "
                  "formulations, against the finest run of the same formulation (dashed) and against the "
                  "transmission-line reference (dotted); grey guides show orders 1 and 2. Expected: the "
                  "self-convergence slopes give the discretisation order; the reference error saturates at "
                  "the linearisation floor (O(Mach, δp/p) ~ 1e-3) of the reference model.")

    # ------------------------------------------------------------------ F5
    def f5_tier2_reflection(self):
        m = self.metrics["tier2"]
        if not m:
            return
        names = [n for n in m["runs"]]
        figure, axis = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
        values = [m["runs"][n]["reflection"]["measured"] for n in names]
        colours = [COLOUR[m["runs"][n]["formulation"]] for n in names]
        axis.bar(range(len(names)), values, color=colours, alpha=0.85)
        first = m["runs"][names[0]]["reflection"]
        axis.axhline(first["reference_solver"], color="black", linestyle="-", label="reference solver (same definition)")
        axis.axhline(first["analytic_front_frequency"], color="grey", linestyle="--", label="analytic Γ(ω_front)")
        axis.axhline(first["analytic_window_frequency"], color="grey", linestyle=":", label="analytic Γ(ω_window)")
        axis.axhline(first["analytic_dc"], color="grey", linestyle="-.", label="analytic Γ(0)")
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("reflection coefficient at 0.75 L1 [-]")
        axis.set_title("Tier 2: measured vs analytic reflection at the return volume")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7)
        self.save(figure, "F5_tier2_reflection",
                  "Tier 2. Reflection coefficient measured on channel 1 at 0.75 L1 (pressure after the "
                  "reflection passes, at 2 L1/c, relative to the incident level at L1/c) for every "
                  "configuration and level, the same measurement on the transmission-line reference (black), "
                  "and the analytic Γ = (Z_vol - Z)/(Z_vol + Z) of the capacitive termination at the front "
                  "frequency, the measurement-window frequency and DC (grey). Expected: the measured value "
                  "converges to the reference-solver value, which lies between the analytic limits because the "
                  "step contains all frequencies. A large offset indicates a wrong port pressure continuity or "
                  "node capacitance.")

    # ------------------------------------------------------------------ F6, F7
    def f6_tier3a_symmetry(self):
        m = self.metrics["tier3a"]
        if not m:
            return
        figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        for name, e in m["runs"].items():
            level = e["level"]
            axis.semilogy(e["S_time"], np.maximum(e["S_subtracted"], 1e-20), color=COLOUR[e["formulation"]],
                          linestyle=":" if e.get("variant", "prop") != "prop" else "--",
                          linewidth=0.8 + 0.3 * ["h", "h2", "h4"].index(level),
                          label=f"{LABEL[e['formulation']]} {level}: max S = {e['max_S_subtracted']:.2e} "
                                f"(raw {e['max_S_raw']:.2e})")
        axis.axhline(1e-10, color="grey", linewidth=0.8, linestyle=":")
        axis.annotate("1e-10", (0.0, 1e-10), fontsize=7, color="grey")
        axis.set_xlabel("t [s]")
        axis.set_ylabel(r"$S(t) = |\dot m(L_2/2, t)| / \max_t |\dot m(0, t)|$ [-]")
        axis.set_title("Tier 3a: spurious midpoint mass flow (background subtracted)")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=6)
        self.save(figure, "F6_tier3a_symmetry",
                  "Tier 3a. Spurious mass flow at the midpoint of the symmetrically heated channel 2, "
                  "relative to the peak expulsion flow at its inlet, with the 1e-3 Pa bias background "
                  "subtracted (raw maxima in the legend), all configurations and levels. Expected: flat at "
                  "round-off (< 1e-10). Anything larger points at a heat-source inconsistency "
                  "(kappa_T q_p - beta q_T != 0) or at an asymmetric friction transformation; the sign of "
                  "mdot(L/2) says towards which port the spurious flow goes.")

    def f7_tier3a_antisymmetry(self):
        runs = self.runs("tier3a")
        m = self.metrics["tier3a"]
        if not runs or not m:
            return
        channel = CHANNELS[1]
        figure, axes = plt.subplots(2, 1, figsize=(7, 5.5), constrained_layout=True, sharex=True,
                                    height_ratios=[2, 1])
        for name, directory in runs.items():
            data = np.load(directory / "state_snapshots.npz")
            e = m["runs"][name]
            t_full = data[f"t_full_{channel}"]
            full = data[f"mdot_full_{channel}"]
            x = data[f"x_{channel}"]
            i = int(np.argmin(np.abs(t_full - e["peak_expulsion_time"])))
            profile = full[i] - full[0]
            style = ":" if e.get("variant", "prop") != "prop" else "--"
            axes[0].plot(x, profile, color=COLOUR[e["formulation"]], linestyle=style,
                         label=f"{name} (t = {t_full[i]:.3f} s)")
            axes[1].plot(x, profile + profile[::-1], color=COLOUR[e["formulation"]], linestyle=style)
        axes[0].set_ylabel(r"$\dot m(x)$ [kg/s]")
        axes[0].set_title("Tier 3a: mass flow at peak expulsion, channel 2")
        axes[1].set_ylabel(r"$\dot m(x) + \dot m(L - x)$ [kg/s]")
        axes[1].set_xlabel("x [m]")
        for axis in axes:
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=6)
        self.save(figure, "F7_tier3a_antisymmetry",
                  "Tier 3a. Mass-flow profile along channel 2 at the time of peak expulsion (background "
                  "subtracted) and, below, the antisymmetry residual mdot(x) + mdot(L - x). Expected: an exactly "
                  "antisymmetric profile and a residual at round-off. A non-zero residual with a definite sign "
                  "indicates a directional bias in the friction or source block.")

    # ------------------------------------------------------------------ F8-F10
    def f8_tier3b_profiles(self):
        runs = self.runs("tier3b")
        m = self.metrics["tier3b"]
        if not runs or not m or not m.get("richardson"):
            return
        channel = CHANNELS[1]
        times = (0.2, 0.5, 1.0)
        figure, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        for column, t in enumerate(times):
            for name, directory in runs.items():
                data = np.load(directory / "state_snapshots.npz")
                manifest = json.loads((directory / "run_manifest.json").read_text())
                ts = data[f"t_snap_{channel}"]
                i = int(np.argmin(np.abs(ts - t)))
                x = data[f"x_{channel}"]
                width = 0.6 + 0.3 * ["h", "h2", "h4"].index(manifest["level"])
                style = ":" if manifest.get("variant", "prop") != "prop" else "--"
                axes[0, column].plot(x, data[f"T_{channel}"][i], color=COLOUR[manifest["formulation"]],
                                     linestyle=style, linewidth=width)
                axes[1, column].plot(x, data[f"mdot_{channel}"][i], color=COLOUR[manifest["formulation"]],
                                     linestyle=style, linewidth=width)
            # Richardson limits are not stored as arrays in metrics.json (size); recompute from the runs.
            for formulation in m["richardson"].keys():
                base, _, variant = formulation.partition("_htc")
                suffix = f"_htc{variant}" if variant else ""
                trio = [runs.get(f"{base}_single_{lvl}{suffix}") for lvl in ("h", "h2", "h4")]
                if any(d is None for d in trio):
                    continue
                datas = [np.load(d / "state_snapshots.npz") for d in trio]
                x = datas[0][f"x_{channel}"]
                for row, name in enumerate(("T", "mdot")):
                    fields = []
                    for data in datas:
                        ts = data[f"t_snap_{channel}"]
                        i = int(np.argmin(np.abs(ts - t)))
                        fields.append(np.interp(x, data[f"x_{channel}"], data[f"{name}_{channel}"][i]))
                    d1 = np.sqrt(np.mean((fields[0] - fields[1]) ** 2))
                    d2 = np.sqrt(np.mean((fields[1] - fields[2]) ** 2))
                    order = np.log2(d1 / d2) if d2 > 0 and d1 > 0 else np.nan
                    limit = fields[2] + (fields[2] - fields[1]) / (2 ** order - 1) if np.isfinite(order) and order > 0 else fields[2]
                    axes[row, column].plot(x, limit, color="black", linewidth=1.0,
                                           linestyle="-" if not variant else ":",
                                           alpha=0.9 if base == "velocity" else 0.5)
            axes[0, column].set_title(f"t = {t:g} s")
            axes[1, column].set_xlabel("x [m]")
        axes[0, 0].set_ylabel("fluid T [K]")
        axes[1, 0].set_ylabel(r"$\dot m$ [kg/s]")
        for axis in axes.ravel():
            axis.grid(alpha=0.3)
        figure.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
        figure.suptitle("Tier 3b: channel 2 temperature and mass flow, all levels, Richardson limits in black")
        self.save(figure, "F8_tier3b_profiles",
                  "Tier 3b. Fluid temperature (top) and mass flow (bottom) along channel 2 at 0.2, 0.5 and "
                  "1.0 s after the asymmetric pulse, both formulations at every level (line width grows with "
                  "refinement), Richardson-extrapolated limits of each formulation in black. Expected: both "
                  "formulations converge to limits that agree within their error estimates. Systematically "
                  "different limits indicate a formulation-dependent modelling difference (e.g. the "
                  "stabilisation block) rather than discretisation error.")

    def f9_tier3b_orders(self):
        m = self.metrics["tier3b"]
        if not m or not m.get("richardson"):
            return
        channel = CHANNELS[1]
        formulations = list(m["richardson"].keys())
        figure, axes = plt.subplots(1, len(formulations), figsize=(4.5 * len(formulations), 4.5),
                                    constrained_layout=True, squeeze=False)
        for axis, formulation in zip(axes[0], formulations):
            rich = m["richardson"][formulation][channel]
            rows, values = [], []
            times = None
            for name in ("mdot", "p", "T"):
                for norm in ("L2", "Linf"):
                    per = rich[name]
                    times = list(per.keys())
                    rows.append(f"{name}, {norm}")
                    values.append([per[t]["orders"][norm]["order"] for t in times])
            values = np.array(values, float)
            image = axis.imshow(values, cmap="coolwarm", vmin=0.0, vmax=2.5, aspect="auto")
            for r in range(values.shape[0]):
                for c in range(values.shape[1]):
                    axis.text(c, r, f"{values[r, c]:.2f}", ha="center", va="center", fontsize=7)
            axis.set_xticks(range(len(times)))
            axis.set_xticklabels([f"{float(t):g} s" for t in times])
            axis.set_yticks(range(len(rows)))
            axis.set_yticklabels(rows)
            axis.set_title(f"{key_style(formulation)[2]}: observed order p̂", fontsize=8)
            figure.colorbar(image, ax=axis, shrink=0.8)
        self.save(figure, "F9_tier3b_orders",
                  "Tier 3b. Observed Richardson order log2(||u_h - u_h/2|| / ||u_h/2 - u_h/4||) per variable "
                  "and norm (rows) and snapshot time (columns), one panel per formulation, channel 2. "
                  "Expected: order ~1 for both formulations (backward-Euler start + upwind), identical between "
                  "formulations. Order < 0.5 or wildly varying values indicate non-asymptotic behaviour "
                  "(under-resolved front) or a level-dependent modelling difference.")

    def f10_tier3b_peak_and_expelled(self):
        m = self.metrics["tier3b"]
        if not m or not m.get("richardson"):
            return
        keys = list(m["richardson"].keys())
        figure, axes = plt.subplots(1, 2, figsize=(5 + 2.5 * len(keys), 4), constrained_layout=True)
        for k, formulation in enumerate(keys):
            colour, linestyle, label = key_style(formulation)
            rich = m["richardson"][formulation]
            peak = rich["peak_strand_temperature"]
            axes[0].errorbar([k], [peak["limit"]], yerr=[peak["error_estimate"]],
                             fmt=formulation_marker(formulation),
                             color=colour, capsize=4, label=f"{label} limit")
            axes[0].plot([k - 0.15, k, k + 0.15], peak["levels"], marker="s", linestyle=linestyle,
                         color=colour, markerfacecolor="none", alpha=0.7)
            expelled = rich["expelled_mass_channel_2"]
            levels = ["h", "h2", "h4"]
            for j, end in enumerate(("inlet", "outlet")):
                values = [expelled[l][end] for l in levels]
                d1, d2 = abs(values[0] - values[1]), abs(values[1] - values[2])
                err = d2 if d2 > 0 else 0.0
                # Formulation by marker shape, port end by fill (inlet filled, outlet open).
                axes[1].errorbar([2 * k + j], [values[2]], yerr=[err],
                                 fmt=formulation_marker(formulation), color=colour,
                                 markerfacecolor=colour if end == "inlet" else "none",
                                 capsize=4, label=f"{label} {end} port")
        axes[0].set_xticks(range(len(keys)))
        axes[0].set_xticklabels(keys, rotation=30, ha="right", fontsize=7)
        axes[0].set_ylabel("peak strand temperature [K]")
        axes[0].set_title("peak T (levels h, h2, h4 and Richardson limit)")
        axes[1].set_xticks(range(2 * len(keys)))
        axes[1].set_xticklabels([f"{key} {end}" for key in keys for end in ("in", "out")],
                                rotation=30, ha="right", fontsize=7)
        axes[1].set_ylabel("expelled mass, channel 2 ports [kg]")
        axes[1].set_title("expelled mass per port (h4, error = |h2 - h4|)")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend(fontsize=6)
        self.save(figure, "F10_tier3b_peak_and_expelled",
                  "Tier 3b. Left: peak strand temperature of channel 2 at the three levels (open squares) and "
                  "its Richardson limit with error bar, both formulations. Right: mass expelled through the "
                  "two ports of channel 2 (integral of the port flow perturbation as passed to the network), "
                  "finest level with the |h2 - h4| difference as error bar. Expected: overlapping error bars "
                  "between formulations.")

    # ------------------------------------------------------------------ F11, F12
    def f11_mass_balance(self):
        m = self.metrics["tier3b"]
        if not m or not m.get("mass_balance"):
            return
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        guides_drawn = False
        for formulation, mb in m["mass_balance"].items():
            colour, linestyle, label = key_style(formulation)
            levels = mb["levels"]
            dts = np.array([levels[l]["time_step"] for l in ("h", "h2", "h4")])
            r = np.array([levels[l]["max_r_port_normalised"] for l in ("h", "h2", "h4")])
            d = np.array([levels[l]["sum_abs_d_int_normalised"] for l in ("h", "h2", "h4")])
            marker = formulation_marker(formulation)
            axes[0].loglog(dts, r, color=colour, linestyle=linestyle, marker=marker,
                           label=f"{label} (slope {mb['slope_r_port']:.2f})")
            axes[1].loglog(dts, d, color=colour, linestyle=linestyle, marker=marker,
                           label=f"{label} (slope {mb['slope_sum_abs_d_int']:.2f})")
            if not guides_drawn:
                if np.all(r > 0):
                    order_guides(axes[0], dts, r.max(), orders=(1, 2))
                if np.all(d > 0):
                    order_guides(axes[1], dts, d.max(), orders=(1, 2))
                guides_drawn = True
        axes[0].set_ylabel(r"$\max_n |r_{port}|$ / expelled mass [1/s]")
        axes[1].set_ylabel(r"$\sum_n |d_{int}|$ / expelled mass [-]")
        for axis in axes:
            axis.set_xlabel(r"$\Delta t$ [s]")
            axis.grid(alpha=0.3, which="both")
            axis.legend(fontsize=7)
        axes[0].set_title("port balance residual")
        axes[1].set_title("interior mass defect")
        figure.suptitle("Mass balance vs time step (Tier 3b)")
        self.save(figure, "F11_mass_balance_scaling",
                  "Tier 3b. Maximum port-balance residual of the two volumes (left) and accumulated interior "
                  "mass defect of the three channels (right), both normalised by the total port excursion, "
                  "versus the time step, both formulations, fitted slopes in the legend, grey order guides. "
                  "Expected (task): r_port first order in dt for the velocity formulation and floored at "
                  "solver tolerance for the mass-flow formulation; d_int of the same order in both. The "
                  "observed slopes are the finding; note that r_port also contains the node-model "
                  "linearisation (C = V rho kappa_T at the old state vs the exact M = V rho(p, T)).")

    def f12_cumulative_defect(self):
        m = self.metrics["tier3b"]
        if not m or not m.get("mass_balance"):
            return
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for formulation, mb in m["mass_balance"].items():
            colour, linestyle, label = key_style(formulation)
            level = mb["levels"].get("h4") or mb["levels"][max(mb["levels"])]
            axis.plot(level["cumulative_time"], level["cumulative_d_int_over_expelled"],
                      color=colour, linestyle=linestyle, label=label)
        axis.set_xlabel("t [s]")
        axis.set_ylabel(r"$\sum_{n} d_{int}(t)$ / expelled mass [-]")
        axis.set_title("Tier 3b: cumulative interior mass defect, finest level")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        self.save(figure, "F12_cumulative_interior_defect",
                  "Tier 3b. Cumulative interior mass defect summed over the three channels, normalised by "
                  "the total port excursion, at the finest level, both formulations. Expected: bounded, "
                  "decreasing with refinement. A defect growing linearly in time indicates a conservation "
                  "error of the interior scheme (the primitive-variable form is not conservative), a jump at "
                  "the pulse indicates the source-consistency issue.")

    # ------------------------------------------------------------------ F13, F14
    TIER4_PANEL_TITLES = {"mdot": r"Mass flow $\dot{m}$", "p": "Pressure $p$",
                          "T": "Temperature $T$", "v": "Velocity $v$"}

    MMS_TIER_LABEL = {"tier4": "Tier 4 (mass flow prescribed, velocity derived)",
                      "tier5": "Tier 5 (velocity prescribed, mass flow derived)"}

    def _tier4_convergence_panels(self, fields, rows, columns, figsize, tier="tier4"):
        """Convergence panels (one per field) of the manufactured-solution
        relative error at t = tau (Tier 4 or Tier 5)."""
        m = self.metrics[tier]
        figure, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        anchor = {}
        for formulation, conv in m["convergence"].items():
            for axis, name in zip(axes, fields):
                c = conv[name]
                dx, err = np.array(c["dx"]), np.array(c["errors"])
                axis.loglog(dx, err, color=COLOUR[formulation], linestyle="--",
                            marker=MARKER[formulation],
                            label=SHORT_LABEL[formulation])
                # Lowest curve value at the second-coarsest level (left corner of the triangle).
                anchor[name] = min(anchor.get(name, np.inf), err[np.argsort(dx)[-2]])
        mesh_sizes = np.array(next(iter(m["convergence"].values()))["mdot"]["dx"])
        for axis, key in zip(axes, fields):
            # First-order convergence triangle just below the curves, right of centre:
            # its bottom-left corner sits a factor 2.2 under the curve at dx_max/2.
            convergence_triangle(axis, mesh_sizes.max(), anchor[key] / 2.2, order=1)
            axis.set_title(self.TIER4_PANEL_TITLES[key], fontsize=10)
            axis.set_xlabel(r"$\Delta x$ (m)")
            axis.set_ylabel(r"Relative error at $t = \tau$ (-)")
            # One labelled tick per level; the automatic minor labels overlap.
            axis.set_xticks(mesh_sizes)
            axis.set_xticklabels([f"{d:g}" for d in mesh_sizes])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="major", linewidth=0.6, alpha=0.6)
            axis.grid(True, which="minor", linewidth=0.4, alpha=0.3)
            axis.legend(fontsize=7 * 1.25, loc="upper left")
        return figure

    def _mms_convergence(self, tier, name):
        m = self.metrics[tier]
        if not m or not m.get("convergence"):
            return
        figure = self._tier4_convergence_panels(("mdot", "p", "T"), 1, 3, (11, 3.8 * 2 / 3), tier=tier)
        self.save(figure, name,
                  f"{self.MMS_TIER_LABEL[tier]}. L2 (root-mean-square over the nodes) of the pointwise "
                  "relative error |analytical - computed| / |analytical| of mass flow, pressure and "
                  "temperature against the manufactured solution at t = tau versus the mesh size (dt "
                  "proportional to dx), both formulations; first-order convergence triangle. Expected: "
                  "identical order in both formulations (first order: upwind diffusion and backward-Euler "
                  "start). A lower order in the mass-flow formulation points at the inherited "
                  "stabilisation coefficients.")

    def _mms_convergence_with_velocity(self, tier, name):
        m = self.metrics[tier]
        if not m or not m.get("convergence") or "v" not in next(iter(m["convergence"].values())):
            return
        figure = self._tier4_convergence_panels(("mdot", "v", "p", "T"), 2, 2, (8, 5), tier=tier)
        derived = ("velocity v_M = mdot_M / (rho(p_M, T_M) A)" if tier == "tier4"
                   else "mass flow mdot_M = rho(p_M, T_M) A v_M")
        self.save(figure, name,
                  f"{self.MMS_TIER_LABEL[tier]}. As the three-panel figure, with the velocity as fourth "
                  f"panel; the derived manufactured field is the {derived}. In the velocity formulation v "
                  "is the primary unknown and mdot = rho A v the derived one; in the mass-flow formulation "
                  "the reverse. Expected: each formulation has the smaller error in its own primary flow "
                  "variable; the derived one adds the density error (from p and T).")

    def f13_tier4_convergence(self):
        self._mms_convergence("tier4", "F13_tier4_mms_convergence")

    def f13b_tier4_convergence_with_velocity(self):
        self._mms_convergence_with_velocity("tier4", "F13b_tier4_mms_convergence_with_velocity")

    def f13c_tier4_error_history(self):
        self._mms_error_history("tier4", "F13c_tier4_error_history")

    def f15_tier5_convergence(self):
        self._mms_convergence("tier5", "F15_tier5_mms_convergence")

    def f15b_tier5_convergence_with_velocity(self):
        self._mms_convergence_with_velocity("tier5", "F15b_tier5_mms_convergence_with_velocity")

    def f15c_tier5_error_history(self):
        self._mms_error_history("tier5", "F15c_tier5_error_history")

    LEVEL_LINESTYLE = {"h": "-", "h2": "--", "h4": "-.", "h8": ":"}
    LEVEL_LABEL = {"h": "$h$", "h2": "$h/2$", "h4": "$h/4$", "h8": "$h/8$"}

    def _mms_error_history(self, tier, name):
        """Relative error of the four fields against time, all levels, both formulations."""
        m = self.metrics[tier]
        if not m or not m.get("runs"):
            return
        fields = ("mdot", "v", "p", "T")
        figure, axes = plt.subplots(2, 2, figsize=(8, 6 * 2 / 3), constrained_layout=True)
        axes = axes.ravel()
        period = next(iter(m["runs"].values())).get("period") or max(
            float(t) for r in m["runs"].values() for t in r["errors"])
        for run in (m["runs"][key] for key in sorted(m["runs"])):
            formulation, level = run["formulation"], run["level"]
            if level not in self.LEVEL_LINESTYLE:
                continue
            times = np.array(sorted(float(t) for t in run["errors"]))
            for axis, key in zip(axes, fields):
                if f"L2_{key}" not in run["errors"][f"{times[0]:.3f}"]:
                    continue
                errors = np.array([run["errors"][f"{t:.3f}"][f"L2_{key}"] for t in times])
                positive = errors > 0  # t = 0 is exact (imposed initial state)
                axis.semilogy(times[positive] / period, errors[positive], color=COLOUR[formulation],
                              linestyle=self.LEVEL_LINESTYLE[level], linewidth=1.2)
        for axis, key in zip(axes, fields):
            axis.set_title(self.TIER4_PANEL_TITLES[key], fontsize=10)
            axis.set_xlabel(r"$t/\tau$ (-)")
            axis.set_ylabel("Relative error (-)")
            axis.set_xlim(0, 1)
            axis.grid(True, which="major", linewidth=0.6, alpha=0.6)
            axis.grid(True, which="minor", linewidth=0.4, alpha=0.3)
        handles = [Line2D([], [], color=COLOUR[f], linestyle="-", label=SHORT_LABEL[f])
                   for f in ("velocity", "mass_flow")]
        handles += [Line2D([], [], color="black", linestyle=ls, label=self.LEVEL_LABEL[level])
                    for level, ls in self.LEVEL_LINESTYLE.items()]
        figure.legend(handles=handles, loc="outside lower center", ncol=6,
                      fontsize=7 * 1.25, frameon=False)
        self.save(figure, name,
                  f"{self.MMS_TIER_LABEL[tier]}. L2 (root-mean-square over the nodes) of the pointwise "
                  "relative error |analytical - computed| / |analytical| of mass flow, velocity, pressure and temperature "
                  "against time over one period of the manufactured solution, for the four refinement "
                  "levels (line style) and both formulations (colour). The error is zero at t = 0 (imposed "
                  "initial state). Expected: growth over the first fraction of the period, then a bounded, "
                  "level-dependent history that halves from level to level (first order); a curve that "
                  "grows without bound indicates an instability or a resonance of the source with the "
                  "discretisation.")

    def f14_picard_and_wall_time(self):
        runs = self.runs("tier3b")
        if not runs:
            return
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        names, walls, colours = [], [], []
        for name, directory in runs.items():
            manifest = json.loads((directory / "run_manifest.json").read_text())
            diagnostics = pd.read_csv(directory / "diagnostics.csv")
            tier = diagnostics[diagnostics["phase"] == "tier"] if "phase" in diagnostics else diagnostics
            axes[0].plot(tier["time_relative"], tier["picard_iters"], color=COLOUR[manifest["formulation"]],
                         linestyle="--", label=name)
            names.append(name)
            walls.append(manifest.get("wall_time_total_s", np.nan))
            colours.append(COLOUR[manifest["formulation"]])
        axes[0].set_xlabel("t [s]")
        axes[0].set_ylabel("Picard iterations per step [-]")
        axes[0].set_title("iteration count (single-pass: 1 by construction)")
        axes[0].set_ylim(0, 2)
        axes[0].legend(fontsize=6)
        axes[1].bar(range(len(names)), walls, color=colours)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("wall time [s]")
        axes[1].set_title("wall time per configuration")
        for axis in axes:
            axis.grid(alpha=0.3)
        self.save(figure, "F14_picard_and_wall_time",
                  "Tier 3b. Left: Picard iteration count per step - identically 1, because the solver has "
                  "no sub-iteration in the thermal-hydraulic step (the iterated mode of the run matrix does "
                  "not exist). Right: wall time per configuration. Expected: wall time scaling with the "
                  "number of steps times the mesh size, similar for both formulations.")

    def all(self):
        for method in (self.f1_tier1_flow_split, self.f2_tier2_pressure_profiles,
                       self.f3_tier2_return_port_flow, self.f4_tier2_convergence,
                       self.f5_tier2_reflection, self.f6_tier3a_symmetry, self.f7_tier3a_antisymmetry,
                       self.f8_tier3b_profiles, self.f9_tier3b_orders, self.f10_tier3b_peak_and_expelled,
                       self.f11_mass_balance, self.f12_cumulative_defect, self.f13_tier4_convergence,
                       self.f13b_tier4_convergence_with_velocity, self.f13c_tier4_error_history,
                       self.f14_picard_and_wall_time, self.f15_tier5_convergence,
                       self.f15b_tier5_convergence_with_velocity, self.f15c_tier5_error_history):
            try:
                method()
            except Exception as error:  # noqa: BLE001
                print(f"[figure] {method.__name__} skipped: {error!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(REPOSITORY_ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(REPOSITORY_ROOT / "figures"))
    args = parser.parse_args(argv)
    Figures(Path(args.results_dir), Path(args.figures_dir)).all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
