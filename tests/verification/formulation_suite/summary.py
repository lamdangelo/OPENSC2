"""results/summary.md writer of the formulation suite (reads metrics.json,
operating_point.json and the run manifests; no solver import)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from formulation_suite.tiers import FORMULATIONS

MDOT_PORT_RULE = (
    "In the velocity formulation the port mass flow passed to the network is "
    "sigma * rho^n * A * v^{n+1}: the Schur coupling block "
    "(hydraulics/network/coupling.py::_coupling_blocks) multiplies the port velocity "
    "unknown by the port-node density of the PREVIOUS step state (node_fields.total_density "
    "as refreshed after step n), and ResolvedPort.mass_flow_into_node, which feeds the node "
    "enthalpy balance, is evaluated after finalize_step but BEFORE the density refresh of "
    "the simulation loop, so it uses the same rho^n with the new v^{n+1}. This is consistent "
    "with the linear system that was solved, but differs from the rho^{n+1} A v^{n+1} stored a "
    "few lines later as node_fields.mass_flow_rate. In the mass-flow formulations the port "
    "slot is the native unknown and sigma * mdot^{n+1} is passed exactly. The node enthalpy "
    "balance uses rho and c_p at the old node state and the new port temperature T^{n+1}."
)


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}e}"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def observed_deviations(metrics: dict) -> list:
    """Lines comparing the measured metrics with the task's 'Expected'
    statements; numbers are pulled from metrics.json."""
    lines = []
    t1 = metrics.get("tier1")
    if t1:
        not_steady = [n for n, e in t1["runs"].items() if not e["stopped_steady"]]
        worst = max((c["eps_mdot_mean"] for e in t1["runs"].values() for c in e["channels"].values()), default=float("nan"))
        lines.append(f"* Tier 1: eps values are NOT at solver tolerance but at the discretisation error of the "
                     f"compressible profile (worst eps_mdot (mean) {fmt(worst)}); whether they decrease with "
                     "refinement is read from the table (a level-independent floor would be a friction-block bias). "
                     + ("All runs reached dW/W < 1e-12" if not not_steady else f"Runs not reaching 1e-12: {not_steady}")
                     + " - only after ~1500 s of thermal settling (frictional/expansion temperature front advected at "
                     "the flow speed), run at dt = 0.5 s; the hydraulic settling alone (~1 s) leaves a 6e-9 per-step drift.")
    t2 = metrics.get("tier2")
    if t2:
        arr = [e["front"]["arrival_error_relative"] for e in t2["runs"].values()]
        lines.append(f"* Tier 2: the channels are friction-dominated at this operating point (F'(v) ~ 45 1/s versus "
                     "omega ~ c/L ~ 10 1/s), so the response to the step is diffusive: the 10 % front arrival at the "
                     f"return volume is 5.5 L1/c, not L1/c (code vs reference relative error {', '.join(fmt(a, 2) for a in arr)}). "
                     "The 'first reflected pulse' and the analytic single-frequency Gamma of the task are not observable "
                     "as such; the reflection metric is the pressure ratio defined in the table, compared between the code "
                     "and the reference solver under the same definition, with the analytic Gamma(omega) limits listed for "
                     "orientation only. L2 errors vs the linear reference sit at the O(Mach, dp/p) ~ 1e-3 linearisation floor.")
    t3a = metrics.get("tier3a")
    if t3a:
        for name, e in t3a["runs"].items():
            lines.append(f"* Tier 3a {name} (heat_transfer_model {e['heat_transfer_model']}): max S = {fmt(e['max_S_subtracted'])}, "
                         f"antisymmetry defect {fmt(e['antisymmetry_defect_subtracted'])}"
                         + (" - NOT at round-off. Diagnosis (experiments on the coarsest velocity run): with the W7-X deck's "
                            "DITTUS_BOELTER_PURE model h -> 0 where |v| -> 0, i.e. exactly at the expansion centre of the heated "
                            "zone; the symmetric state is then unstable to symmetry breaking during the pulse (the centre "
                            "displaces by ~1.5 elements, the strand temperature becomes asymmetric by 0.18 K) and the midpoint "
                            "flow reaches 20-25 % of the expulsion flow in BOTH formulations, decaying to ~1e-5 after the pulse. "
                            "With the floored Dittus-Boelter model (heat_transfer_model 1, variant htc1) the same case is "
                            "symmetric to ~2e-9, which is the 1e-3 Pa bias flow. This is a heat-transfer-model instability, "
                            "not a heat-source consistency or friction-transformation defect (both formulations identical)."
                            if e["heat_transfer_model"] == 2 and e["max_S_subtracted"] > 1e-6 else
                            " (the residual is the 1e-3 Pa bias flow: background mdot(L/2)/max expulsion = "
                            f"{fmt(abs(e['background_mass_flow_mid']) / e['max_expulsion_flow'])})."))
    t3b = metrics.get("tier3b")
    if t3b and t3b.get("mass_balance"):
        for key, mb in t3b["mass_balance"].items():
            lines.append(f"* Tier 3b mass balance ({key}): fitted slope of max|r_port| vs dt = {fmt(mb['slope_r_port'], 2)}, "
                         f"of sum|d_int| vs dt = {fmt(mb['slope_sum_abs_d_int'], 2)} (expected: first order for the velocity "
                         "formulation's r_port, floored for the mass-flow one; same order for d_int). Note r_port also "
                         "contains the node linearisation C = V rho kappa_T at the old state vs the exact M_i = V rho(p_i, T_i).")
    for tier, label in (("tier4", "Tier 4"), ("tier5", "Tier 5")):
        t = metrics.get(tier)
        if t and t.get("convergence"):
            for formulation, conv in t["convergence"].items():
                velocity = f", v {fmt(conv['v']['order'], 2)}" if "v" in conv else ""
                lines.append(f"* {label} {formulation}: observed orders mdot {fmt(conv['mdot']['order'], 2)}, p {fmt(conv['p']['order'], 2)}, "
                             f"T {fmt(conv['T']['order'], 2)}{velocity} (expected: identical between formulations).")
    return lines


def write_summary(results_root: Path) -> Path:
    results_root = Path(results_root)
    point = load(results_root / "operating_point.json")
    metrics = {tier: load(results_root / tier / "metrics.json")
               for tier in ("tier1", "tier2", "tier3a", "tier3b", "tier4", "tier5")}
    manifests = {}
    for tier in metrics:
        tier_root = results_root / tier
        if tier_root.exists():
            for directory in sorted(tier_root.iterdir()):
                manifest = load(directory / "run_manifest.json")
                if manifest:
                    manifests[(tier, directory.name)] = manifest
    lines = ["# Formulation verification suite - summary", ""]
    first = next(iter(manifests.values()), {})
    lines += [
        "## Setup",
        "",
        f"* Conductor definition: `{point['conductor_template']}` (W7-X NbTi CICC), one bundle "
        f"channel A = {point['channel']['cross_section']:.4e} m^2, D_h = "
        f"{point['channel']['hydraulic_diameter']:.4e} m, void fraction {point['channel']['void_fraction']}, "
        f"friction model {point['channel']['friction_factor_model']} (multiplier "
        f"{point['channel']['friction_multiplier']}); strand NbTi-W7X/Cu, Al6063 jacket, glass-epoxy "
        "insulation; no electric problem (current mode none).",
        f"* Channel lengths 20 / 30 / 40 m; volumes 0.5 L; far-side linear valves R_far = "
        f"{point['far_resistance']:.4e} Pa/(kg/s) = 1.0 Z_line (Z_line = c/A = {point['impedance_Pa_s_per_kg']:.4e}); "
        f"reservoirs {point['supply_reservoir']:.1f} / {point['return_reservoir']:.1f} Pa so that the "
        f"volumes sit at {point['pressure_supply_volume']:.1f} / {point['pressure_return_volume']:.1f} Pa.",
        f"* Property library: CoolProp {first.get('property_library', {}).get('CoolProp', '?')} "
        f"(git {first.get('property_library', {}).get('CoolProp_git', '?')}), tabulated property path "
        f"{first.get('property_library', {}).get('tabulated_properties', '?')}; code git hash "
        f"{first.get('git_hash', '?')}.",
        "* Iteration mode: single-pass only. The thermal-hydraulic step has no Picard sub-iteration "
        "(one linear solve per step, coefficients at the old level); `picard_iters = 1` in every "
        "diagnostics.csv. The 'iterated Picard' axis of the run matrix is therefore not available "
        "without solver changes.",
        "",
        "### |v|/c per channel (Tier-1 reference state)",
        "",
        "| channel | mdot [kg/s] | max |v|/c |",
        "|---|---|---|",
    ]
    for channel, mach in point["mach"].items():
        k = int(channel[-1]) - 1
        lines.append(f"| {channel} | {point['mass_flows'][k]:.4e} | {mach:.3e} |")
    lines += ["", "### Port mass-flow evaluation rule (velocity formulation)", "", MDOT_PORT_RULE, ""]

    # Tier 1
    t1 = metrics["tier1"]
    if t1:
        lines += ["## Tier 1 - steady flow split", "",
                  f"Reference flows {', '.join(f'{m:.6e}' for m in t1['reference']['mass_flows'])} kg/s, "
                  f"dp = {t1['reference']['pressure_drop']:.4f} Pa.", "",
                  "| run | steady? | final dW/W | channel | eps_mdot (mean) | eps_mdot (max) | axial var. | eps_dp | eps_v (max) | eps_p profile | eps_T [K] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, entry in t1["runs"].items():
            for channel, c in entry["channels"].items():
                lines.append(
                    f"| {name} | {entry['stopped_steady']} | {fmt(entry['final_steady_change'])} | {channel} | "
                    f"{fmt(c['eps_mdot_mean'])} | {fmt(c['eps_mdot_max'])} | {fmt(c['mdot_axial_variation'])} | "
                    f"{fmt(c['eps_dp'])} | {fmt(c['eps_v_max'])} | {fmt(c['eps_p_profile_max'])} | {fmt(c['eps_T_profile_max'])} |")
        lines.append("")
    # Tier 2
    t2 = metrics["tier2"]
    if t2:
        lines += ["## Tier 2 - low-Mach acoustic transient", "",
                  f"Transmission-line reference: {t2['reference']['points_per_metre']:.0f} points/m, "
                  f"refinement change {fmt(t2['reference']['refinement_change'])} of the step.", "",
                  "| run | dt | elements | arrival (code / ref) | p_return at 1.5 L1/c (code / ref) | Gamma meas / ref-solver / analytic(front, window, dc) |",
                  "|---|---|---|---|---|---|"]
        for name, e in t2["runs"].items():
            f = e["front"]
            r = e["reflection"]
            lines.append(
                f"| {name} | {fmt(e['time_step'])} | {e['elements']} | {fmt(f['arrival_time'], 5)} / {fmt(f['arrival_time_reference'], 5)} | "
                f"{fmt(f['return_pressure_at_1.5_transit'])} / {fmt(f['return_pressure_reference_at_1.5_transit'])} | "
                f"{fmt(r['measured'])} / {fmt(r['reference_solver'])} / ({fmt(r['analytic_front_frequency'])}, {fmt(r['analytic_window_frequency'])}, {fmt(r['analytic_dc'])}) |")
        lines += ["", "L2 errors vs the transmission-line reference (normalised by the step / step/Z):", "",
                  "| run | channel | t/(L1/c) | L2 p | L2 mdot | Linf p |", "|---|---|---|---|---|---|"]
        for name, e in t2["runs"].items():
            for channel, errs in e["errors"].items():
                for t, v in errs.items():
                    lines.append(f"| {name} | {channel} | {t} | {fmt(v['L2_p'])} | {fmt(v['L2_mdot'])} | {fmt(v['Linf_p'])} |")
        lines += ["", "Observed orders (self-convergence against the finest run; reference error slope in brackets):", "",
                  "| formulation | series | levels | order p | order mdot | (ref p) | (ref mdot) |", "|---|---|---|---|---|---|---|"]
        for formulation, conv in t2["convergence"].items():
            for series, c in conv.items():
                lines.append(f"| {formulation} | {series} | {','.join(c['levels'])} | {fmt(c['order_self_p'], 2)} | "
                             f"{fmt(c['order_self_mdot'], 2)} | {fmt(c['order_reference_p'], 2)} | {fmt(c['order_reference_mdot'], 2)} |")
        lines.append("")
    # Tier 3a
    t3a = metrics["tier3a"]
    if t3a:
        lines += ["## Tier 3a - symmetric heating", "",
                  "| run | mesh symmetry | node at L/2 | background mdot(L/2) | max expulsion mdot | max S raw | max S subtracted | antisym. raw | antisym. subtracted | peak strand T |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, e in t3a["runs"].items():
            T = e["max_strand_temperature"].get("CONDUCTOR_2", {}).get("T") if e.get("max_strand_temperature") else None
            lines.append(f"| {name} | {fmt(e['mesh_symmetry_defect'])} | {e['midpoint_is_exact']} | {fmt(e['background_mass_flow_mid'])} | "
                         f"{fmt(e['max_expulsion_flow'])} | {fmt(e['max_S_raw'])} | {fmt(e['max_S_subtracted'])} | "
                         f"{fmt(e['antisymmetry_defect_raw'])} | {fmt(e['antisymmetry_defect_subtracted'])} | {fmt(T, 4)} |")
        lines.append("")
    # Tier 3b
    t3b = metrics["tier3b"]
    if t3b and t3b.get("richardson"):
        lines += ["## Tier 3b - asymmetric heating with through-flow (Richardson)", "",
                  "| formulation | channel | variable | time | order L2 | order Linf | L2 err vs limit (h, h2, h4) |",
                  "|---|---|---|---|---|---|---|"]
        for formulation, rich in t3b["richardson"].items():
            for channel in ("CHAN1_C1", "CHAN1_C2", "CHAN1_C3"):
                for name, per in rich.get(channel, {}).items():
                    for t, v in per.items():
                        lines.append(f"| {formulation} | {channel} | {name} | {t} | {fmt(v['orders']['L2']['order'], 2)} | "
                                     f"{fmt(v['orders']['Linf']['order'], 2)} | {', '.join(fmt(e) for e in v['L2_error_vs_limit'])} |")
            peak = rich.get("peak_strand_temperature", {})
            lines += ["", f"{formulation}: peak strand T levels {', '.join(f'{T:.4f}' for T in peak.get('levels', []))} K, "
                          f"order {fmt(peak.get('order'), 2)}, limit {fmt(peak.get('limit'), 5)} +- {fmt(peak.get('error_estimate'))} K; "
                          f"expelled mass channel 2 (inlet/outlet, h4): "
                          f"{peak and fmt(rich['expelled_mass_channel_2']['h4']['inlet'])} / {fmt(rich['expelled_mass_channel_2']['h4']['outlet'])} kg.", ""]
        lines += ["Cross-check of the extrapolated limits (velocity vs mass-flow, |diff| <= sum of error estimates):", "",
                  "| variant: channel | variable | time | max diff | max budget | fraction of nodes within budget |", "|---|---|---|---|---|---|"]
        for variant, per_channel in t3b.get("cross_check", {}).items():
            for channel, per_name in per_channel.items():
                for name, per in per_name.items():
                    for t, v in per.items():
                        lines.append(f"| {variant}: {channel} | {name} | {t} | {fmt(v['max_diff'])} | {fmt(v['max_budget'])} | {v['fraction_within_budget']:.3f} |")
        lines += ["", "### Mass balance vs dt (normalised by the total port excursion)", "",
                  "| formulation | level | dt | max r_port | sum d_int | sum |d_int| |", "|---|---|---|---|---|---|"]
        for formulation, mb in t3b.get("mass_balance", {}).items():
            for level, v in mb["levels"].items():
                lines.append(f"| {formulation} | {level} | {fmt(v['time_step'])} | {fmt(v['max_r_port_normalised'])} | "
                             f"{fmt(v['sum_d_int_normalised'])} | {fmt(v['sum_abs_d_int_normalised'])} |")
            lines.append(f"| {formulation} | slope vs dt | | {fmt(mb['slope_r_port'], 2)} | | {fmt(mb['slope_sum_abs_d_int'], 2)} |")
        lines.append("")
    # Tiers 4 and 5 (manufactured solutions)
    for tier, heading in (("tier4", "## Tier 4 - manufactured solution (mass flow prescribed, velocity derived)"),
                          ("tier5", "## Tier 5 - manufactured solution (velocity prescribed, mass flow derived)")):
        t = metrics.get(tier)
        if not t:
            continue
        lines += [heading, "",
                  "Error measure: pointwise relative error |analytical - computed| / |analytical| at every node, "
                  "then the L2 (root-mean-square) norm over the nodes, at t = tau.", "",
                  "| formulation | variable | dx | L2 of relative error | order |", "|---|---|---|---|---|"]
        for formulation, conv in t["convergence"].items():
            for name, v in conv.items():
                lines.append(f"| {formulation} | {name} | {', '.join(fmt(d, 2) for d in v['dx'])} | "
                             f"{', '.join(fmt(e) for e in v['errors'])} | {fmt(v['order'], 2)} |")
        lines.append("")
    # Wall time
    lines += ["## Wall time", "", "| tier | run | steps | wall [s] | per step [ms] |", "|---|---|---|---|---|"]
    for (tier, name), m in manifests.items():
        lines.append(f"| {tier} | {name} | {m.get('steps_total')} | {fmt(m.get('wall_time_total_s'), 1)} | "
                     f"{fmt(1e3 * (m.get('wall_time_per_step_s') or float('nan')), 1)} |")
    lines += ["", "## Deviations from expectation", "",
              "Pre-known (design):",
              "* No Picard sub-iteration exists in the solver; the iterated mode of the run matrix is not executed.",
              "* The node capacitance of the code is C = V rho kappa_T (isothermal, node state), not V/c^2; the "
              "transmission-line reference uses the code's definition.",
              "* R_far = 1.0 Z_line instead of the planned 10 Z_line (the latter would leave 665 Pa of the 0.5 bar "
              "drop to the channels and Mach 1.5e-4); reservoir pressures offset by R_far Q.",
              "* A node at L_2/2 requires an EVEN element count (the task says odd).",
              "* Zero initial flow crashes the channel (friction 10/Re at Re = 0 gives inf * 0 = NaN, no floor on the "
              "channel path): Tier 3a uses a 1e-3 Pa return-side bias and reports S(t) raw and background-subtracted.",
              "* The Tier-1 reference is the compressible, isenthalpic-type steady ODE of the code's model (density "
              "varies by ~1 % over the channel), not the scalar Darcy-Weisbach equation.",
              "* The Tier-2 reservoir 'step' is a C-infinity smooth step of 4 ms (0.9 m front) applied identically "
              "in the code runs and in the reference; the line reference includes the mean-flow advection, the "
              "tangent of the code's friction law (R' = F'(v)/A, not 2 f |v|/D_h) and the base pressure gradient.",
              "* The Tier-4 boundary conditions are inlet mass flow + outlet pressure + inflow temperature "
              "(all constant in time for this manufactured solution), imposed through the existing "
              "hydraulic_boundary_condition 3, not Dirichlet on every variable at both ends.",
              "",
              "Observed:"]
    lines += observed_deviations(metrics)
    lines.append("")
    path = results_root / "summary.md"
    path.write_text("\n".join(lines))
    print(f"[summary] {path}")
    return path
