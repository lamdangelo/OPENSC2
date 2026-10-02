"""Metrics of the formulation suite, evaluated from the run files alone
(results/<tier>/<run>/{state_snapshots.npz, port_timeseries.csv,
diagnostics.csv, run_manifest.json}) - no solver import. Writes
results/<tier>/metrics.json (and results/tier2/reference/*.npz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from formulation_suite import geometry
from formulation_suite.tiers import FORMULATIONS, LEVEL_FACTORS, operating_point
from reference.transmission_line import (
    LineBase,
    NodeTermination,
    reflection_coefficient_volume,
    smooth_step,
    solve_transmission_line,
)

CHANNELS = ("CHAN1_C1", "CHAN1_C2", "CHAN1_C3")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class Run:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "run_manifest.json").read_text())
        self.snapshots = np.load(self.directory / "state_snapshots.npz")
        self.ports = pd.read_csv(self.directory / "port_timeseries.csv")
        self.diagnostics = pd.read_csv(self.directory / "diagnostics.csv")
        self.formulation = self.manifest["formulation"]
        self.level = self.manifest["level"]
        self.variant = self.manifest.get("variant", "prop")

    @property
    def name(self) -> str:
        return self.directory.name

    def x(self, channel):
        return self.snapshots[f"x_{channel}"]

    def field(self, channel, name, index=-1):
        return self.snapshots[f"{name}_{channel}"][index]

    def snapshot_times(self, channel=CHANNELS[0]):
        return self.snapshots[f"t_snap_{channel}"]

    def tier_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "phase" in frame.columns:
            return frame[frame["phase"] == "tier"]
        return frame[frame["time_relative"] >= 0.0]


def load_runs(tier_root: Path) -> list:
    runs = []
    for directory in sorted(tier_root.iterdir()):
        if directory.is_dir() and (directory / "run_manifest.json").exists():
            runs.append(Run(directory))
    return runs


def l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def linf(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def rel(a, b):
    return float(abs(a - b) / abs(b)) if b != 0 else float("nan")


def pointwise_relative_error(computed, analytical):
    """|analytical - computed| / |analytical| at every node."""
    computed, analytical = np.asarray(computed, dtype=float), np.asarray(analytical, dtype=float)
    return np.abs(analytical - computed) / np.abs(analytical)


def l2_relative(computed, analytical):
    """L2 (root-mean-square) norm of the pointwise relative error."""
    return float(np.sqrt(np.mean(pointwise_relative_error(computed, analytical) ** 2)))


def linf_relative(computed, analytical):
    """Maximum of the pointwise relative error."""
    return float(pointwise_relative_error(computed, analytical).max())


def fitted_order(steps, errors) -> float:
    steps, errors = np.asarray(steps, float), np.asarray(errors, float)
    mask = errors > 0
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(steps[mask]), np.log(errors[mask]), 1)
    return float(slope)


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------


def tier1_metrics(runs: list) -> dict:
    point = operating_point()
    reference = point.reference
    out = {"reference": {"mass_flows": [float(m) for m in reference.mass_flows],
                         "pressure_drop": reference.pressure_supply_volume - reference.pressure_return_volume,
                         "pressure_supply_volume": reference.pressure_supply_volume,
                         "pressure_return_volume": reference.pressure_return_volume},
           "runs": {}}
    for run in runs:
        entry = {"formulation": run.formulation, "level": run.level,
                 "stopped_steady": run.manifest.get("stopped_steady"),
                 "final_steady_change": run.manifest.get("final_steady_change"),
                 "wall_time_s": run.manifest.get("wall_time_total_s"),
                 "steps": run.manifest.get("steps_total"), "channels": {}}
        for k, channel in enumerate(CHANNELS):
            profile = reference.profiles[k]
            x = run.x(channel)
            mdot = run.field(channel, "mdot")
            p = run.field(channel, "p")
            v = run.field(channel, "v")
            T = run.field(channel, "T")
            v_ref = np.interp(x, profile.x, profile.velocity)
            p_ref = np.interp(x, profile.x, profile.pressure)
            T_ref = np.interp(x, profile.x, profile.temperature)
            m_ref = float(reference.mass_flows[k])
            drop = float(p[0] - p[-1])
            entry["channels"][channel] = {
                "mass_flow_mean": float(mdot.mean()),
                "mass_flow_inlet": float(mdot[0]),
                "eps_mdot_mean": rel(float(mdot.mean()), m_ref),
                "eps_mdot_max": float(np.abs(mdot - m_ref).max() / m_ref),
                "mdot_axial_variation": float(np.ptp(mdot) / m_ref),
                "eps_dp": rel(drop, out["reference"]["pressure_drop"]),
                "eps_v_max": float(np.abs(v - v_ref).max() / np.abs(v_ref).max()),
                "eps_v_inlet": rel(float(v[0]), float(v_ref[0])),
                "eps_p_profile_max": float(np.abs(p - p_ref).max() / out["reference"]["pressure_drop"]),
                "eps_T_profile_max": float(np.abs(T - T_ref).max()),
                "direct_quantity": "mass_flow" if run.formulation == "mass_flow" else "velocity",
            }
        # Port flows as passed vs the channel fields (last row).
        last = run.ports.iloc[-1]
        entry["port_flows_as_passed"] = {
            channel: {"inlet": float(last[f"mdot_port_inlet_{channel}"]),
                      "outlet": float(last[f"mdot_port_outlet_{channel}"])}
            for channel in CHANNELS
        }
        entry["volume_pressures"] = {"supply": float(last["p_supply_volume"]),
                                     "return": float(last["p_return_volume"])}
        entry["eps_volume_pressure"] = {
            "supply": rel(float(last["p_supply_volume"]), reference.pressure_supply_volume),
            "return": rel(float(last["p_return_volume"]), reference.pressure_return_volume),
        }
        out["runs"][run.name] = entry
    return out


# --------------------------------------------------------------------------
# Tier 2
# --------------------------------------------------------------------------


def tier2_reference(results_root: Path, manifest: dict, points_per_metre: int = 100,
                    tolerance: float = 1.0e-6) -> dict:
    """Transmission-line reference of the Tier-2 case, self-refined; cached
    in results/tier2/reference/."""
    cache = results_root / "tier2" / "reference"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "transmission_line.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    point = operating_point()
    reference = point.reference
    inputs = point.inputs
    friction = point.friction
    duration = manifest["duration"]
    snapshot_times = manifest["snapshot_times"]
    node_props = geometry.helium_properties(geometry.TEMPERATURE, reference.pressure_supply_volume)
    node_props_r = geometry.helium_properties(geometry.TEMPERATURE, reference.pressure_return_volume)
    capacitance_s = geometry.VOLUME * node_props["density"] * node_props["isothermal_compressibility"]
    capacitance_r = geometry.VOLUME * node_props_r["density"] * node_props_r["isothermal_compressibility"]

    def build_lines(density_of_points):
        lines = []
        for spec, profile in zip(point.channel_specs, reference.profiles):
            n = int(round(spec.length * density_of_points)) + 1
            x = np.linspace(0.0, spec.length, n)
            p = np.interp(x, profile.x, profile.pressure)
            T = np.interp(x, profile.x, profile.temperature)
            props = geometry.helium_properties(T, p)
            rho = props["density"]
            c = props["speed_of_sound"]
            mu = props["viscosity"]
            phi = props["isobaric_expansion_coefficient"] / (rho * props["isothermal_compressibility"] * props["isochoric_specific_heat"])
            v = float(reference.mass_flows[list(point.channel_specs).index(spec)]) / (rho * spec.cross_section)

            def acceleration(vel):
                reynolds = np.abs(vel) * rho * spec.hydraulic_diameter / mu
                return 2.0 * np.asarray(friction(reynolds)) * np.abs(vel) * vel / spec.hydraulic_diameter

            h = 1.0e-6 * np.abs(v)
            F = acceleration(v)
            F_tangent = (acceleration(v + h) - acceleration(v - h)) / (2 * h)
            G_tangent = (
                acceleration(v + h) * phi * rho * (v + h) - acceleration(v - h) * phi * rho * (v - h)
            ) / (2 * h)
            dp_dx = np.gradient(p, x)
            pressure_work = -dp_dx - G_tangent
            lines.append(LineBase(spec.identifier, x, spec.cross_section, rho, c, v, F_tangent,
                                  pressure_work))
        return lines

    def terminations():
        total = reference.total_mass_flow
        gain = node_props_r["isobaric_expansion_coefficient"] * geometry.TEMPERATURE / (
            node_props_r["density"] * node_props_r["isothermal_compressibility"]
            * node_props_r["isochoric_specific_heat"]) / (node_props_r["density"] * node_props_r["speed_of_sound"] ** 2) * geometry.TEMPERATURE
        supply = NodeTermination(capacitance_s, point.far_resistance,
                                 reservoir=smooth_step(manifest["step_amplitude"], manifest["step_ramp"]),
                                 thermal=True, volume=geometry.VOLUME, density=node_props["density"],
                                 expansion_coefficient=node_props["isobaric_expansion_coefficient"],
                                 inflow_rate=total, inflow_temperature_gain=0.0)
        return_node = NodeTermination(capacitance_r, point.far_resistance, thermal=True,
                                      volume=geometry.VOLUME, density=node_props_r["density"],
                                      expansion_coefficient=node_props_r["isobaric_expansion_coefficient"],
                                      inflow_rate=total, inflow_temperature_gain=gain)
        return supply, return_node

    previous = None
    density_of_points = points_per_metre
    result = None
    change = np.nan
    for _ in range(3):
        lines = build_lines(density_of_points)
        supply, return_node = terminations()
        result = solve_transmission_line(lines, supply, return_node, duration, snapshot_times,
                                         cfl=0.1, series_stride=1)
        if previous is not None:
            change = 0.0
            for line in lines:
                x_prev = previous.snapshots[line.identifier]["x"]
                for i in range(len(snapshot_times)):
                    p_new = result.snapshots[line.identifier]["pressure"][i]
                    p_old = np.interp(line.x, x_prev, previous.snapshots[line.identifier]["pressure"][i])
                    change = max(change, l2(p_new, p_old) / manifest["step_amplitude"])
            if change < tolerance:
                break
        previous = result
        density_of_points *= 2
    payload = {"times": result.times, "node_pressure_supply": result.node_pressure_supply,
               "node_pressure_return": result.node_pressure_return,
               "snapshot_times": result.snapshot_times, "refinement_change": np.array(change),
               "points_per_metre": np.array(density_of_points / 2), "time_step": np.array(result.time_step),
               "capacitance_supply": np.array(capacitance_s), "capacitance_return": np.array(capacitance_r)}
    for line in lines:
        payload[f"x_{line.identifier}"] = result.snapshots[line.identifier]["x"]
        payload[f"p_{line.identifier}"] = result.snapshots[line.identifier]["pressure"]
        payload[f"mdot_{line.identifier}"] = result.snapshots[line.identifier]["mass_flow"]
        payload[f"port_in_{line.identifier}"] = result.port_flow_inlet[line.identifier]
        payload[f"port_out_{line.identifier}"] = result.port_flow_outlet[line.identifier]
    np.savez_compressed(path, **payload)
    return payload


def _steady_baseline(run: Run, channel: str, name: str):
    """Pre-step baseline (settled state) of a field from the diagnostics:
    the first tier snapshot is at 0.5 L1/c; the settled profile is recovered
    from the port time series at t_rel = 0 for scalar quantities, and for
    profiles from the Tier-1 reference (the settled state)."""
    return None


def tier2_metrics(runs: list, results_root: Path) -> dict:
    point = operating_point()
    reference_solution = point.reference
    prop_runs = [r for r in runs if r.variant == "prop"]
    if not runs:
        return {}
    manifest = runs[0].manifest
    ref = tier2_reference(results_root, manifest)
    transit_1 = manifest["transit_time_channel_1"]
    step = manifest["step_amplitude"]
    out = {"reference": {"points_per_metre": float(ref["points_per_metre"]),
                         "refinement_change": float(ref["refinement_change"]),
                         "capacitance_supply": float(ref["capacitance_supply"])},
           "runs": {}, "convergence": {}}
    # Baseline (settled) profiles: the Tier-1 reference profiles.
    baseline = {}
    for k, (spec, profile) in enumerate(zip(point.channel_specs, reference_solution.profiles)):
        baseline[spec.identifier] = profile
    for run in runs:
        entry = {"formulation": run.formulation, "level": run.level, "variant": run.variant,
                 "time_step": run.manifest["time_step"], "elements": run.manifest["elements"],
                 "wall_time_s": run.manifest.get("wall_time_total_s"), "errors": {}}
        t_snap = run.snapshot_times()
        for channel in CHANNELS:
            x = run.x(channel)
            # Baseline: the run's own settled state (snapshot at t_rel = 0).
            p0 = run.snapshots[f"p_{channel}"][0]
            m0 = run.snapshots[f"mdot_{channel}"][0]
            errors = {}
            for i, t in enumerate(t_snap):
                if t <= 0.0:
                    continue
                j = int(np.argmin(np.abs(ref["snapshot_times"] - t)))
                if abs(ref["snapshot_times"][j] - t) > 0.05 * transit_1:
                    continue
                x_ref = ref[f"x_{channel}"]
                dp_ref = np.interp(x, x_ref, ref[f"p_{channel}"][j])
                dm_ref = np.interp(x, x_ref, ref[f"mdot_{channel}"][j])
                dp = run.snapshots[f"p_{channel}"][i] - p0
                dm = run.snapshots[f"mdot_{channel}"][i] - m0
                label = f"{t / transit_1:.2f}"
                errors[label] = {"L2_p": l2(dp, dp_ref) / step, "L2_mdot": l2(dm, dm_ref) / (step / point.impedance),
                                 "Linf_p": linf(dp, dp_ref) / step}
            entry["errors"][channel] = errors
        # Front arrival at the return volume: 10 % of the step.
        tier = run.tier_rows(run.ports)
        t_rel = tier["time_relative"].to_numpy()
        p_r = tier["p_return_volume"].to_numpy() - tier["p_return_volume"].to_numpy()[0]
        arrival = t_rel[np.argmax(p_r > 0.1 * step)] if np.any(p_r > 0.1 * step) else np.nan
        ref_p_r = ref["node_pressure_return"]
        ref_arrival = ref["times"][np.argmax(ref_p_r > 0.1 * step)] if np.any(ref_p_r > 0.1 * step) else np.nan
        # Amplitude at the return volume: value at 1.05 L2/c (after channel 1's front, before channel 2's).
        t_amp = 1.5 * transit_1
        entry["front"] = {
            "arrival_time": float(arrival), "arrival_time_reference": float(ref_arrival),
            "arrival_error_relative": rel(float(arrival), float(ref_arrival)),
            "return_pressure_at_1.5_transit": float(np.interp(t_amp, t_rel, p_r)),
            "return_pressure_reference_at_1.5_transit": float(np.interp(t_amp, ref["times"], ref_p_r)),
        }
        # Reflection coefficient of channel 1 at x = 0.75 L1.
        x1 = run.x(CHANNELS[0])
        # incident amplitude: snapshot at 1.0 transit (front passed 0.75 L1 at 0.75 transit)
        i_inc = int(np.argmin(np.abs(t_snap - 1.0 * transit_1)))
        probe = int(np.argmin(np.abs(x1 - 0.75 * geometry.LENGTHS[0])))
        p_base = run.snapshots[f"p_{CHANNELS[0]}"][0][probe]
        p_inc = run.snapshots[f"p_{CHANNELS[0]}"][i_inc][probe] - p_base
        i_ref = int(np.argmin(np.abs(t_snap - 2.0 * transit_1)))
        p_after = run.snapshots[f"p_{CHANNELS[0]}"][i_ref][probe] - p_base
        j_inc = int(np.argmin(np.abs(ref["snapshot_times"] - 1.0 * transit_1)))
        j_ref = int(np.argmin(np.abs(ref["snapshot_times"] - 2.0 * transit_1)))
        xr = ref[f"x_{CHANNELS[0]}"]
        pr_inc = np.interp(x1[probe], xr, ref[f"p_{CHANNELS[0]}"][j_inc])
        pr_after = np.interp(x1[probe], xr, ref[f"p_{CHANNELS[0]}"][j_ref])
        omega_front = 2 * np.pi / (2 * manifest["step_ramp"])
        omega_window = 2 * np.pi / (4 * 0.5 * transit_1)
        entry["reflection"] = {
            "measured": float((p_after - p_inc) / p_inc) if p_inc != 0 else np.nan,
            "reference_solver": float((pr_after - pr_inc) / pr_inc) if pr_inc != 0 else np.nan,
            "analytic_front_frequency": complex(reflection_coefficient_volume(
                point.impedance, point.far_resistance, float(ref["capacitance_return"]), omega_front)).real,
            "analytic_window_frequency": complex(reflection_coefficient_volume(
                point.impedance, point.far_resistance, float(ref["capacitance_return"]), omega_window)).real,
            "analytic_dc": complex(reflection_coefficient_volume(
                point.impedance, point.far_resistance, float(ref["capacitance_return"]), 1e-9)).real,
            "definition": "(p(0.75 L1, 2 L1/c) - p(0.75 L1, L1/c)) / p(0.75 L1, L1/c), channel 1",
        }
        out["runs"][run.name] = entry
    # Convergence: error vs the finest run of the same formulation and vs the reference,
    # last common snapshot (4 L1/c).
    for formulation in FORMULATIONS:
        conv = {}
        for series in ("prop", "dx", "dt"):
            members = [r for r in runs if r.formulation == formulation and
                       (r.variant == series or (series != "prop" and r.variant == "prop" and r.level == "h4"))]
            members = sorted(members, key=lambda r: LEVEL_FACTORS[r.level])
            if len(members) < 2:
                continue
            finest = members[-1]
            steps, err_p, err_m, ref_p, ref_m = [], [], [], [], []
            i_last = int(np.argmin(np.abs(finest.snapshot_times() - 4.0 * transit_1)))
            for run in members:
                i_run = int(np.argmin(np.abs(run.snapshot_times() - 4.0 * transit_1)))
                e_p = e_m = r_p = r_m = 0.0
                for channel in CHANNELS:
                    x = run.x(channel)
                    xf = finest.x(channel)
                    p_run = run.snapshots[f"p_{channel}"][i_run]
                    m_run = run.snapshots[f"mdot_{channel}"][i_run]
                    p_fin = np.interp(x, xf, finest.snapshots[f"p_{channel}"][i_last])
                    m_fin = np.interp(x, xf, finest.snapshots[f"mdot_{channel}"][i_last])
                    e_p = max(e_p, l2(p_run, p_fin) / step)
                    e_m = max(e_m, l2(m_run, m_fin) / (step / point.impedance))
                    j = int(np.argmin(np.abs(ref["snapshot_times"] - 4.0 * transit_1)))
                    dp_ref = np.interp(x, ref[f"x_{channel}"], ref[f"p_{channel}"][j])
                    dm_ref = np.interp(x, ref[f"x_{channel}"], ref[f"mdot_{channel}"][j])
                    r_p = max(r_p, l2(p_run - run.snapshots[f"p_{channel}"][0], dp_ref) / step)
                    r_m = max(r_m, l2(m_run - run.snapshots[f"mdot_{channel}"][0], dm_ref) / (step / point.impedance))
                size = (geometry.LENGTHS[0] / run.manifest["elements"][0]) if series != "dt" else run.manifest["time_step"]
                steps.append(size)
                err_p.append(e_p)
                err_m.append(e_m)
                ref_p.append(r_p)
                ref_m.append(r_m)
            conv[series] = {
                "levels": [r.level for r in members], "step_sizes": steps,
                "self_error_p": err_p, "self_error_mdot": err_m,
                "reference_error_p": ref_p, "reference_error_mdot": ref_m,
                "order_self_p": fitted_order(steps[:-1], err_p[:-1]),
                "order_self_mdot": fitted_order(steps[:-1], err_m[:-1]),
                "order_reference_p": fitted_order(steps, ref_p),
                "order_reference_mdot": fitted_order(steps, ref_m),
            }
        out["convergence"][formulation] = conv
    return out


# --------------------------------------------------------------------------
# Tier 3a
# --------------------------------------------------------------------------


def tier3a_metrics(runs: list) -> dict:
    out = {"runs": {}}
    channel = CHANNELS[1]
    for run in runs:
        x = run.x(channel)
        L = float(x[-1])
        mid = int(np.argmin(np.abs(x - 0.5 * L)))
        mesh_symmetry = float(np.abs(x + x[::-1] - L).max() / L)
        full = run.snapshots[f"mdot_full_{channel}"]
        t_full = run.snapshots[f"t_full_{channel}"]
        background = full[0]  # settled (bias) flow at the start of the tier phase
        inlet = np.abs(full[:, 0] - background[0]).max()
        S_raw = np.abs(full[:, mid]) / max(inlet, 1e-300)
        S_sub = np.abs(full[:, mid] - background[mid]) / max(inlet, 1e-300)
        anti = full + full[:, ::-1]
        anti_sub = (full - background) + (full - background)[:, ::-1]
        scale = np.abs(full - background).max()
        i_peak = int(np.argmax(np.abs(full[:, 0] - background[0])))
        out["runs"][run.name] = {
            "formulation": run.formulation, "level": run.level, "variant": run.variant,
            "heat_transfer_model": run.manifest.get("heat_transfer_model", 2),
            "elements": run.manifest["elements"], "time_step": run.manifest["time_step"],
            "mesh_symmetry_defect": mesh_symmetry, "midpoint_node_x": float(x[mid]),
            "midpoint_is_exact": bool(abs(x[mid] - 0.5 * L) < 1e-12 * L),
            "background_mass_flow_mid": float(background[mid]),
            "background_mass_flow_inlet": float(background[0]),
            "max_expulsion_flow": float(inlet),
            "max_S_raw": float(S_raw.max()), "max_S_subtracted": float(S_sub.max()),
            "antisymmetry_defect_raw": float(np.abs(anti).max() / scale),
            "antisymmetry_defect_subtracted": float(np.abs(anti_sub).max() / scale),
            "peak_expulsion_time": float(t_full[i_peak]),
            "max_strand_temperature": run.manifest.get("max_strand_temperature"),
            "S_time": t_full.tolist()[:: max(1, len(t_full) // 400)],
            "S_raw": S_raw.tolist()[:: max(1, len(t_full) // 400)],
            "S_subtracted": S_sub.tolist()[:: max(1, len(t_full) // 400)],
            "wall_time_s": run.manifest.get("wall_time_total_s"),
        }
    return out


# --------------------------------------------------------------------------
# Tier 3b: Richardson
# --------------------------------------------------------------------------


def richardson(u_h, u_h2, u_h4):
    """Observed order and extrapolated limit from three levels (fields on
    the common coarse grid)."""
    d1 = u_h - u_h2
    d2 = u_h2 - u_h4
    out = {}
    for norm, fn in (("L2", lambda d: np.sqrt(np.mean(d ** 2))), ("Linf", lambda d: np.abs(d).max())):
        n1, n2 = fn(d1), fn(d2)
        order = float(np.log2(n1 / n2)) if n2 > 0 and n1 > 0 else float("nan")
        out[norm] = {"order": order, "diff_h_h2": float(n1), "diff_h2_h4": float(n2)}
    order = out["L2"]["order"]
    if np.isfinite(order) and order > 0:
        limit = u_h4 + (u_h4 - u_h2) / (2 ** order - 1)
        error_estimate = np.abs(u_h4 - u_h2) / (2 ** order - 1)
    else:
        limit = u_h4.copy()
        error_estimate = np.abs(u_h4 - u_h2)
    return out, limit, error_estimate


def tier3b_metrics(runs: list) -> dict:
    out = {"runs": {}, "richardson": {}, "cross_check": {}, "mass_balance": {}}
    by = {(r.formulation, r.variant, r.level): r for r in runs}
    variants = sorted({r.variant for r in runs}, key=lambda v: (v != "prop", v))
    limits = {}
    keys = [(f, v) for v in variants for f in FORMULATIONS]
    for formulation_base, variant in keys:
        formulation = formulation_base if variant == "prop" else f"{formulation_base}_{variant}"
        trio = [by.get((formulation_base, variant, level)) for level in ("h", "h2", "h4")]
        if any(r is None for r in trio):
            continue
        rh, rh2, rh4 = trio
        fields = {}
        for channel in CHANNELS:
            x = rh.x(channel)
            fields[channel] = {}
            for name in ("mdot", "p", "T", "Tstrand"):
                per_time = {}
                for i, t in enumerate(rh.snapshot_times()):
                    def at(run):
                        j = int(np.argmin(np.abs(run.snapshot_times() - t)))
                        return np.interp(x, run.x(channel), run.snapshots[f"{name}_{channel}"][j])
                    orders, limit, err = richardson(at(rh), at(rh2), at(rh4))
                    per_time[f"{t:.2f}"] = {"orders": orders, "limit": limit, "error": err,
                                            "L2_error_vs_limit": [l2(at(r), limit) for r in trio],
                                            "Linf_error_vs_limit": [linf(at(r), limit) for r in trio]}
                fields[channel][name] = per_time
        limits[formulation] = fields
        out["richardson"][formulation] = {
            channel: {name: {t: {"orders": v["orders"],
                                 "L2_error_vs_limit": v["L2_error_vs_limit"],
                                 "Linf_error_vs_limit": v["Linf_error_vs_limit"]}
                             for t, v in per.items()}
                      for name, per in fields[channel].items()}
            for channel in CHANNELS
        }
        # Peak strand temperature with Richardson error bar.
        peaks = [r.manifest["max_strand_temperature"][f"CONDUCTOR_2"] for r in trio]
        T = np.array([p["T"] for p in peaks])
        t = np.array([p["t_relative"] for p in peaks])
        d1, d2 = abs(T[0] - T[1]), abs(T[1] - T[2])
        order = float(np.log2(d1 / d2)) if d2 > 0 else float("nan")
        err = d2 / (2 ** order - 1) if np.isfinite(order) and order > 0 else d2
        out["richardson"][formulation]["peak_strand_temperature"] = {
            "levels": T.tolist(), "times": t.tolist(), "order": order,
            "limit": float(T[2] + (T[2] - T[1]) / (2 ** order - 1)) if np.isfinite(order) and order > 0 else float(T[2]),
            "error_estimate": float(err)}
        # Expelled mass per port of channel 2.
        expelled = {}
        for run in trio:
            tier = run.tier_rows(run.ports)
            tt = tier["time_relative"].to_numpy()
            values = {}
            for end in ("inlet", "outlet"):
                flow = tier[f"mdot_port_{end}_{CHANNELS[1]}"].to_numpy()
                values[end] = float(np.trapezoid(flow - flow[0], tt))
            total = 0.0
            for channel in CHANNELS:
                for end in ("inlet", "outlet"):
                    flow = tier[f"mdot_port_{end}_{channel}"].to_numpy()
                    total += float(np.trapezoid(np.abs(flow - flow[0]), tt))
            values["total_excursion_all_ports"] = total
            expelled[run.level] = values
        out["richardson"][formulation]["expelled_mass_channel_2"] = expelled
        # Mass balance vs dt (normalised by the total port excursion).
        balance = {}
        for run in trio:
            tier = run.tier_rows(run.diagnostics)
            total = expelled[run.level]["total_excursion_all_ports"]
            r_max = max(float(np.nanmax(np.abs(tier[f"r_port_{node}"]))) for node in ("supply_volume", "return_volume"))
            d_sum = sum(float(np.nansum(tier[f"d_int_{channel}"])) for channel in CHANNELS)
            d_abs = sum(float(np.nansum(np.abs(tier[f"d_int_{channel}"]))) for channel in CHANNELS)
            balance[run.level] = {"time_step": run.manifest["time_step"],
                                  "max_r_port_normalised": r_max / total if total else np.nan,
                                  "sum_d_int_normalised": d_sum / total if total else np.nan,
                                  "sum_abs_d_int_normalised": d_abs / total if total else np.nan,
                                  "expelled_mass": total,
                                  "cumulative_d_int_over_expelled": (
                                      np.nancumsum(sum(tier[f"d_int_{c}"].to_numpy() for c in CHANNELS)) / total
                                  ).tolist()[:: max(1, len(tier) // 400)],
                                  "cumulative_time": tier["time_relative"].to_numpy().tolist()[:: max(1, len(tier) // 400)]}
        dts = [balance[l]["time_step"] for l in ("h", "h2", "h4")]
        out["mass_balance"][formulation] = {
            "levels": balance,
            "slope_r_port": fitted_order(dts, [balance[l]["max_r_port_normalised"] for l in ("h", "h2", "h4")]),
            "slope_sum_abs_d_int": fitted_order(dts, [balance[l]["sum_abs_d_int_normalised"] for l in ("h", "h2", "h4")]),
        }
    # Cross-formulation check of the extrapolated limits (per variant).
    for variant in variants:
        pair = ["velocity", "mass_flow"] if variant == "prop" else [f"velocity_{variant}", f"mass_flow_{variant}"]
        if not all(f in limits for f in pair):
            continue
        checks = {}
        for channel in CHANNELS:
            checks[channel] = {}
            for name in ("mdot", "p", "T"):
                per = {}
                for t, v_vel in limits[pair[0]][channel][name].items():
                    v_mf = limits[pair[1]][channel][name].get(t)
                    if v_mf is None:
                        continue
                    diff = np.abs(v_vel["limit"] - v_mf["limit"])
                    budget = v_vel["error"] + v_mf["error"]
                    per[t] = {"max_diff": float(diff.max()), "max_budget": float(budget.max()),
                              "within_budget_everywhere": bool(np.all(diff <= budget + 1e-15 * np.abs(v_vel["limit"]).max())),
                              "fraction_within_budget": float(np.mean(diff <= budget + 1e-15 * np.abs(v_vel["limit"]).max()))}
                checks[channel][name] = per
        out["cross_check"][variant] = checks
    for run in runs:
        out["runs"][run.name] = {"formulation": run.formulation, "level": run.level,
                                 "variant": run.variant,
                                 "time_step": run.manifest["time_step"], "elements": run.manifest["elements"],
                                 "wall_time_s": run.manifest.get("wall_time_total_s"),
                                 "steps": run.manifest.get("steps_total"),
                                 "max_strand_temperature": run.manifest.get("max_strand_temperature")}
    return out


# --------------------------------------------------------------------------
# Tier 4
# --------------------------------------------------------------------------


def tier4_metrics(runs: list, tier: str = "tier4") -> dict:
    """Manufactured-solution errors; ``tier`` selects the solution (Tier 4:
    mass flow prescribed, Tier 5: velocity prescribed)."""
    from formulation_suite.tiers import manufactured_solution, manufactured_solution_velocity
    solution = manufactured_solution() if tier == "tier4" else manufactured_solution_velocity()
    out = {"runs": {}, "convergence": {},
           "prescribed_flow_variable": "mass_flow" if tier == "tier4" else "velocity"}
    channel = "CHAN1_C1"
    area = geometry.channel_inputs(geometry.load_template()).cross_section

    def manufactured_velocity(x, t):
        # Derived field v_M = mdot_M / (rho(p_M, T_M) A), same property library as the solver.
        density = geometry.helium_properties(solution.temperature(x, t), solution.pressure(x, t))["density"]
        return solution.mass_flow(x, t) / (density * area)

    for run in runs:
        x = run.x(channel)
        errors = {}
        for i, t in enumerate(run.snapshot_times(channel)):
            # Pointwise relative error |analytical - computed| / |analytical|,
            # then the L2 (RMS) or L-infinity norm over the nodes.
            exact = {"mdot": solution.mass_flow(x, t), "p": solution.pressure(x, t),
                     "T": solution.temperature(x, t), "v": manufactured_velocity(x, t)}
            entry = {}
            for name, analytical in exact.items():
                computed = run.snapshots[f"{name}_{channel}"][i]
                entry[f"L2_{name}"] = l2_relative(computed, analytical)
                entry[f"Linf_{name}"] = linf_relative(computed, analytical)
            errors[f"{t:.3f}"] = entry
        out["runs"][run.name] = {"formulation": run.formulation, "level": run.level,
                                 "elements": run.manifest["elements"], "time_step": run.manifest["time_step"],
                                 "period": solution.period,
                                 "errors": errors, "wall_time_s": run.manifest.get("wall_time_total_s")}
    for formulation in FORMULATIONS:
        members = sorted([r for r in runs if r.formulation == formulation],
                         key=lambda r: LEVEL_FACTORS[r.level])
        if len(members) < 2:
            continue
        dx = [solution.length / r.manifest["elements"][0] for r in members]
        final = {}
        for name in ("mdot", "p", "T", "v"):
            errs = [out["runs"][r.name]["errors"][f"{solution.period:.3f}"][f"L2_{name}"] for r in members]
            final[name] = {"dx": dx, "errors": errs, "order": fitted_order(dx, errs)}
        out["convergence"][formulation] = final
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def evaluate_tier(tier: str, results_root: Path) -> dict:
    tier_root = results_root / tier
    runs = load_runs(tier_root)
    if tier == "tier1":
        metrics = tier1_metrics(runs)
    elif tier == "tier2":
        metrics = tier2_metrics(runs, results_root)
    elif tier == "tier3a":
        metrics = tier3a_metrics(runs)
    elif tier == "tier3b":
        metrics = tier3b_metrics(runs)
    elif tier in ("tier4", "tier5"):
        metrics = tier4_metrics(runs, tier=tier)
    else:
        raise ValueError(tier)
    (tier_root / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json_default))
    print(f"[metrics] {tier}: {len(runs)} runs -> {tier_root / 'metrics.json'}")
    return metrics


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return value.real
    return str(value)
