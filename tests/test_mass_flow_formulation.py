"""Integration tests of the selectable (mdot, p, T) hydraulic formulation.

Covers, on the shortened CASE_1 scenario (see test_network_coupling):

* the byte-identity guard: a deck without the new key and a deck with an
  explicit ``hydraulic_formulation: velocity`` produce identical Output
  trees (the mass-flow machinery is provably inert on the velocity path);
* the auto-resolution rules on real runs (uncoupled -> velocity, network
  section declared -> mass_flow, explicit velocity + coupling -> warning);
* cross-formulation agreement on a heated pressure-BC transient: pressure
  and temperature agree closely; the flow fields carry the measured,
  refinement-invariant continuous difference of the frozen-M conjugation
  route (the omitted coefficient-derivative terms - see the bound block
  below and docs/mass_flow_formulation_notes.md), asserted as bounded and
  refinement-stable, with exact agreement treated as suspicious;
* the mass-conservation drift of both formulations on a flow-reversal
  transient (reported, not assumed);
* the steady-state axial mass-flow uniformity of both formulations
  (discretization-order prediction, checked by mesh refinement).

Deviation bounds are pinned from measured values with a documented margin,
following the test_network_coupling convention.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hydraulics.hydraulic_flags import HydraulicFormulation
from simulation import Simulation

from test_network_coupling import (
    NETWORK_SECTION,
    SHORTENED_END_TIME,
    boost_heat_pulse,
    impose_pressure_drop_on_both_channels,
    prepare_run_directory,
    run_simulation,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def set_hydraulic_formulation(run_directory: Path, value: str) -> None:
    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    document = yaml.safe_load(conductor_file.read_text())
    document["conductor"]["inputs"]["hydraulic_formulation"] = value
    conductor_file.write_text(yaml.safe_dump(document, sort_keys=False))


def formulation_edit(value: str):
    def edit(conductor_document) -> None:
        conductor_document["conductor"]["inputs"]["hydraulic_formulation"] = value

    return edit


def compose_edits(*edits):
    def edit(conductor_document) -> None:
        for single_edit in edits:
            single_edit(conductor_document)

    return edit


def grid_elements_edit(number_of_elements: int):
    def edit(conductor_document) -> None:
        conductor_document["grid"]["number_of_elements"] = number_of_elements

    return edit


def heating_off_edit(conductor_document) -> None:
    for component in conductor_document["components"]:
        operations = component.get("operations", {})
        if "heat_flux_mode" in operations:
            operations["heat_flux_mode"] = 0


def gentle_heat_pulse_edit(conductor_document) -> None:
    """The retimed pulse at 100 W/m: a genuine heated transient with the
    flow staying forward everywhere. Used for the cross-formulation
    agreement check, which needs a REGULAR transient: near the backflow
    threshold the trajectory becomes so parameter-sensitive that the
    velocity-vs-mass-flow comparison measures chaos amplification instead
    of discretization-order operator differences (measured O(1) deviations
    growing under refinement at 300 W/m, while the same comparison sits at
    1e-6 in the near-steady coupled limit and below 5e-3 in steady state)."""
    boost_heat_pulse(conductor_document)
    for component in conductor_document["components"]:
        operations = component.get("operations", {})
        if "heat_flux_amplitude" in operations:
            operations["heat_flux_amplitude"] = 100.0


def moderated_heat_pulse_edit(conductor_document) -> None:
    """The retimed committed pulse at 300 W/m: strong enough to fully
    reverse both inlet flows (measured min inlet flows -18.6 / -30.9 g/s
    against +8.4 / +12.5 g/s nominal), while staying below the ~500 W/m
    threshold where the pre-existing velocity path loses the inlet
    temperature boundary condition under the backflow sign flip (invalid
    negative temperature at t = 1 s, present already on the unmodified
    velocity formulation)."""
    boost_heat_pulse(conductor_document)
    for component in conductor_document["components"]:
        operations = component.get("operations", {})
        if "heat_flux_amplitude" in operations:
            operations["heat_flux_amplitude"] = 300.0


def output_tree_hashes(run_directory: Path) -> dict:
    output_roots = list(run_directory.glob("*/*/Output"))
    assert len(output_roots) == 1, output_roots
    return {
        str(path.relative_to(output_roots[0])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output_roots[0].rglob("*"))
        if path.is_file()
    }


def fluid_final_fields(simulation: Simulation) -> dict:
    """{channel: {field: array}} of the final nodal state."""
    conductor = simulation.list_of_Conductors[0]
    return {
        f_comp.identifier: {
            field_name: getattr(
                f_comp.coolant.node_fields, field_name
            ).copy()
            for field_name in (
                "velocity",
                "pressure",
                "temperature",
                "mass_flow_rate",
            )
        }
        for f_comp in conductor.inventory.fluids.collection
    }


def relative_deviation(reference: np.ndarray, other: np.ndarray) -> float:
    return float(
        np.abs(other - reference).max() / np.abs(reference).max()
    )


def channel_cross_sections(run_directory: Path) -> dict:
    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    document = yaml.safe_load(conductor_file.read_text())
    return {
        component["identifier"]: float(component["inputs"]["cross_section"])
        for component in document["components"]
        if component["kind"] == "CHAN"
    }


def output_directory(run_directory: Path) -> Path:
    output_roots = list(run_directory.glob("*/*/Output"))
    assert len(output_roots) == 1
    return output_roots[0]


def linear_mass_inventory(profile_file: Path, cross_section: float) -> float:
    """integral rho A dz from a Solution/Initialization profile file."""
    frame = pd.read_csv(profile_file, sep="\t")
    zcoord = frame["zcoord (m)"].to_numpy()
    density = frame["total_density (kg/m^3)"].to_numpy()
    return float(np.trapezoid(density * cross_section, zcoord))


def boundary_flow_integrals(te_file: Path) -> tuple:
    """(integral mdot_inl dt, integral mdot_out dt, min mdot_inl).

    Right-endpoint rectangle rule: the backward-Euler update advances the
    inventory with the new-time boundary flows, so this quadrature is the
    one consistent with the scheme (a trapezoid here would charge the
    measurement itself with O(dt * dmdot/dt) error, which dominates during
    the flow reversal)."""
    frame = pd.read_csv(te_file, sep="\t")
    time = frame["time (s)"].to_numpy()
    inlet = frame["mass_flow_rate_inl (kg/s)"].to_numpy()
    outlet = frame["mass_flow_rate_out (kg/s)"].to_numpy()
    steps = np.diff(time)
    return (
        float(np.sum(inlet[1:] * steps)),
        float(np.sum(outlet[1:] * steps)),
        float(inlet.min()),
    )


# --------------------------------------------------------------------------
# Byte-identity guard and configuration resolution
# --------------------------------------------------------------------------


def test_default_matches_explicit_velocity_byte_identically(tmp_path):
    default_directory = prepare_run_directory(tmp_path, "default", end_time=1.0)
    default_run = run_simulation(default_directory)

    explicit_directory = prepare_run_directory(
        tmp_path, "explicit", end_time=1.0
    )
    set_hydraulic_formulation(explicit_directory, "velocity")
    explicit_run = run_simulation(explicit_directory)

    for simulation in (default_run, explicit_run):
        assert (
            simulation.list_of_Conductors[0].hydraulic_formulation
            is HydraulicFormulation.VELOCITY
        )

    default_hashes = output_tree_hashes(default_directory)
    explicit_hashes = output_tree_hashes(explicit_directory)
    differing = sorted(
        name
        for name in set(default_hashes) | set(explicit_hashes)
        if default_hashes.get(name) != explicit_hashes.get(name)
    )
    assert not differing, (
        "default (absent key) and explicit velocity runs differ in: "
        f"{differing}"
    )


def test_auto_resolves_to_mass_flow_with_network(tmp_path):
    coupled = run_simulation(
        prepare_run_directory(
            tmp_path, "auto_coupled", network_section=NETWORK_SECTION,
            end_time=0.5,
        )
    )
    conductor = coupled.list_of_Conductors[0]
    assert conductor.hydraulic_formulation is HydraulicFormulation.MASS_FLOW
    for f_comp in conductor.inventory.fluids.collection:
        assert (
            f_comp.coolant.hydraulic_formulation
            is HydraulicFormulation.MASS_FLOW
        )


def test_explicit_velocity_with_network_warns_and_wins(tmp_path):
    run_directory = prepare_run_directory(
        tmp_path, "velocity_coupled", network_section=NETWORK_SECTION,
        end_time=0.5,
    )
    set_hydraulic_formulation(run_directory, "velocity")
    with pytest.warns(UserWarning, match="lagged-density"):
        coupled = run_simulation(run_directory)
    assert (
        coupled.list_of_Conductors[0].hydraulic_formulation
        is HydraulicFormulation.VELOCITY
    )


# --------------------------------------------------------------------------
# Cross-formulation agreement (heated pressure-BC transient)
# --------------------------------------------------------------------------

# Measured cross-formulation deviations (gentle 100 W/m pulse, 2 s,
# final state) with a ~3x margin: pressure 6.4e-4, temperature 1.2-2.1e-2,
# flow (v and mdot) 1.2-1.4e-1. The flow deviation is a KNOWN CONTINUOUS
# difference of the frozen-M conjugation route, not a discretization
# error: the exact change of variables W = M(W) W' produces
# coefficient-derivative terms M^-1 (dM/dt + A dM/dx) W' that the frozen
# similarity omits. They vanish in (near-)steady states - the coupled
# degenerate limit agrees to 1e-6 and the steady axial uniformity to
# <5e-3 - but during density transients they are O(v * drho/rho):
# measured invariant under BOTH mesh refinement (50 -> 100 elements) AND
# time-step refinement (0.1 -> 0.025 s), which is the fingerprint of an
# omitted continuous term rather than an O(h) or O(dt) effect. See
# docs/mass_flow_formulation_notes.md.
AGREEMENT_BOUNDS = {
    "pressure": 2.0e-3,
    "temperature": 6.0e-2,
    "velocity": 3.0e-1,
    "mass_flow_rate": 3.0e-1,
}
# The deviations must be refinement-STABLE (the continuous difference is
# mesh-independent; growth under refinement would signal an instability).
REFINEMENT_STABILITY_FACTOR = 1.5


def cross_formulation_deviations(tmp_path, number_of_elements: int) -> dict:
    edits = compose_edits(
        impose_pressure_drop_on_both_channels,
        gentle_heat_pulse_edit,
        grid_elements_edit(number_of_elements),
    )
    runs = {}
    for formulation in ("velocity", "mass_flow"):
        runs[formulation] = run_simulation(
            prepare_run_directory(
                tmp_path,
                f"{formulation}_{number_of_elements}",
                conductor_edits=compose_edits(
                    edits, formulation_edit(formulation)
                ),
            )
        )
    velocity_fields = fluid_final_fields(runs["velocity"])
    mass_flow_fields = fluid_final_fields(runs["mass_flow"])
    return {
        (channel, field_name): relative_deviation(
            velocity_fields[channel][field_name],
            mass_flow_fields[channel][field_name],
        )
        for channel in velocity_fields
        for field_name in velocity_fields[channel]
    }


def test_cross_formulation_agreement_and_convergence(tmp_path):
    coarse = cross_formulation_deviations(tmp_path, 50)
    fine = cross_formulation_deviations(tmp_path, 100)

    report = "\n".join(
        f"  {channel} {field_name}: 50 el {coarse[channel, field_name]:.3e}"
        f" -> 100 el {fine[channel, field_name]:.3e}"
        for channel, field_name in sorted(coarse)
    )
    print(f"\ncross-formulation deviations:\n{report}")

    for (channel, field_name), deviation in coarse.items():
        bound = AGREEMENT_BOUNDS[field_name]
        assert deviation < bound, (
            f"{channel} {field_name}: cross-formulation deviation "
            f"{deviation:.3e} exceeds {bound:.0e}\n{report}"
        )
        # Exact agreement would be suspicious: the formulations differ.
        assert deviation > 0.0
    for (channel, field_name), coarse_deviation in coarse.items():
        fine_deviation = fine[channel, field_name]
        assert fine_deviation < REFINEMENT_STABILITY_FACTOR * max(
            coarse_deviation, 1.0e-12
        ), (
            f"{channel} {field_name}: cross-formulation deviation GROWS "
            f"under mesh refinement ({coarse_deviation:.3e} -> "
            f"{fine_deviation:.3e})\n{report}"
        )


# --------------------------------------------------------------------------
# Mass-conservation drift (flow-reversal transient)
# --------------------------------------------------------------------------

# Relative drift bound pinned from measurement with ~2.5x margin: with the
# BE-consistent boundary quadrature the measured drifts on the 300 W/m
# reversal scenario are 6.19e-2 (velocity) and 3.24e-2 (mass_flow),
# relative to the integrated inlet throughput. These are genuine
# conservation defects of the non-conservative primitive-variable scheme
# (pressure row = EOS form + upwind diffusion), not measurement noise; the
# mass-flow formulation halves the defect here (asserted below with the
# measured ~1.9x separation).
DRIFT_BOUND = 1.5e-1


def mass_conservation_drift(tmp_path, formulation: str) -> tuple:
    """(relative drift, min inlet flow) of a strong-pulse run."""
    run_directory = prepare_run_directory(
        tmp_path,
        f"drift_{formulation}",
        conductor_edits=compose_edits(
            impose_pressure_drop_on_both_channels,
            moderated_heat_pulse_edit,
            formulation_edit(formulation),
        ),
    )
    run_simulation(run_directory)
    cross_sections = channel_cross_sections(run_directory)
    output_root = output_directory(run_directory)

    total_boundary_gain = 0.0
    total_inventory_change = 0.0
    total_throughput = 0.0
    minimum_inlet_flow = np.inf
    for channel, cross_section in cross_sections.items():
        inlet_integral, outlet_integral, minimum_inlet = (
            boundary_flow_integrals(
                output_root
                / "Time_evolution"
                / "CONDUCTOR_1"
                / f"{channel}_inlet_outlet_te.tsv"
            )
        )
        total_boundary_gain += inlet_integral - outlet_integral
        total_throughput += abs(inlet_integral)
        minimum_inlet_flow = min(minimum_inlet_flow, minimum_inlet)
        initial_inventory = linear_mass_inventory(
            output_root / "Initialization" / "CONDUCTOR_1" / f"{channel}.tsv",
            cross_section,
        )
        final_inventory = linear_mass_inventory(
            output_root / "Solution" / "CONDUCTOR_1" / f"{channel}.tsv",
            cross_section,
        )
        total_inventory_change += final_inventory - initial_inventory

    relative_drift = abs(total_boundary_gain - total_inventory_change) / (
        total_throughput
    )
    return relative_drift, minimum_inlet_flow


def test_mass_conservation_drift_under_flow_reversal(tmp_path):
    drifts = {
        formulation: mass_conservation_drift(tmp_path, formulation)
        for formulation in ("velocity", "mass_flow")
    }
    report = "; ".join(
        f"{formulation}: drift {drift:.3e} (min mdot_inl {min_inl:.3e} kg/s)"
        for formulation, (drift, min_inl) in drifts.items()
    )
    print(f"\nmass-conservation drift: {report}")

    for formulation, (drift, minimum_inlet_flow) in drifts.items():
        # The pulse must actually reverse the inlet flow, or the test
        # is not exercising the advertised scenario.
        assert minimum_inlet_flow < 0.0, (
            f"{formulation}: no inlet backflow observed ({report})"
        )
        assert drift < DRIFT_BOUND, (
            f"{formulation}: mass-conservation drift {drift:.3e} exceeds "
            f"{DRIFT_BOUND:.0e} ({report})"
        )
    # Measured ordering (not assumed): the native-mdot formulation halves
    # the conservation defect on this scenario (3.24e-2 vs 6.19e-2).
    assert drifts["mass_flow"][0] < drifts["velocity"][0], (
        f"mass_flow formulation no longer conserves better ({report})"
    )


# --------------------------------------------------------------------------
# Steady-state axial mass-flow uniformity
# --------------------------------------------------------------------------

STEADY_END_TIME = 3.0
# Pinned from measurement with margin; see test output for actual values.
STEADY_VARIATION_BOUND = 5.0e-3


def steady_mass_flow_variation(tmp_path, formulation: str,
                               number_of_elements: int) -> float:
    run_directory = prepare_run_directory(
        tmp_path,
        f"steady_{formulation}_{number_of_elements}",
        end_time=STEADY_END_TIME,
        conductor_edits=compose_edits(
            impose_pressure_drop_on_both_channels,
            heating_off_edit,
            grid_elements_edit(number_of_elements),
            formulation_edit(formulation),
        ),
    )
    simulation = run_simulation(run_directory)
    conductor = simulation.list_of_Conductors[0]
    variation = 0.0
    for f_comp in conductor.inventory.fluids.collection:
        mass_flow = f_comp.coolant.node_fields.mass_flow_rate
        variation = max(
            variation,
            float(
                np.abs(mass_flow - mass_flow.mean()).max()
                / abs(mass_flow.mean())
            ),
        )
    return variation


def test_steady_state_mass_flow_uniformity(tmp_path):
    variations = {
        (formulation, elements): steady_mass_flow_variation(
            tmp_path, formulation, elements
        )
        for formulation in ("velocity", "mass_flow")
        for elements in (50, 100)
    }
    report = "\n".join(
        f"  {formulation} {elements} el: {variation:.3e}"
        for (formulation, elements), variation in sorted(variations.items())
    )
    print(f"\nsteady axial mass-flow variation:\n{report}")

    for (formulation, elements), variation in variations.items():
        if variation < 1.0e-10:
            # Machine-precision uniformity: possible in principle if the
            # discrete continuity row telescopes exactly; documented as an
            # explicit outcome rather than silently passing the bound.
            continue
        assert variation < STEADY_VARIATION_BOUND, (
            f"{formulation} {elements} el: steady mass-flow variation "
            f"{variation:.3e} exceeds {STEADY_VARIATION_BOUND:.0e}\n{report}"
        )
    for formulation in ("velocity", "mass_flow"):
        coarse = variations[formulation, 50]
        fine = variations[formulation, 100]
        if coarse > 1.0e-10:
            assert fine < coarse, (
                f"{formulation}: steady variation does not shrink under "
                f"refinement\n{report}"
            )
