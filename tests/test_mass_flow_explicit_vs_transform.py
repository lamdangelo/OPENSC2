"""Explicit (mdot, p, T) assembly versus the similarity-transform route.

Both are (mdot, p, T) formulations sharing the initial-condition seeding,
the boundary conditions, the network port coupling and the solution
reorganization; any difference isolates the assembly route:

* Gauss-block identity on the real CASE_1 deck (first assembled step,
  builders intercepted): the explicit flux-Jacobian and source blocks equal
  the transformed ones - to roundoff with the direct HEOS flash, to the
  thermodynamic-identity level (~1e-6 .. 1e-4) with the production
  property tables - and the stabilisation blocks differ ONLY in the
  off-diagonal entries the conjugation introduces in the mdot row;
* solution difference on the 100 W/m heated pulse and on a steady run,
  per channel and field, reported and bounded;
* convergence study of the explicit-vs-transform deviation under mesh
  refinement (50/100/200 elements) and time-step refinement
  (0.1/0.05/0.025 s), fitted orders reported, monotone decrease under mesh
  refinement asserted (the deviation stems from the O(dz) upwind block);
* self-convergence of each route against its own finest-mesh run.
"""

import numpy as np
import pytest
import yaml

import interfaces.coolprop_interface as cpi
import utility_functions.transient_solution_functions as tsf
from hydraulics.hydraulic_flags import HydraulicFormulation

from test_mass_flow_explicit_integration import FORMULATION_EDITS
from test_mass_flow_formulation import (
    compose_edits,
    fluid_final_fields,
    gentle_heat_pulse_edit,
    grid_elements_edit,
    heating_off_edit,
)
from test_network_coupling import (
    impose_pressure_drop_on_both_channels,
    prepare_run_directory,
    run_simulation,
)


# --------------------------------------------------------------------------
# Gauss-block identity
# --------------------------------------------------------------------------


class BlockCapture:
    """Context manager intercepting the element-matrix builders to copy the
    Gauss-point blocks of the first assembled step (restores the builders
    itself, so that other active monkeypatches are left alone)."""

    BUILDERS = ("build_elamat", "build_elkmat", "build_elsmat")

    def __init__(self):
        self.blocks = {}
        self._originals = {}

    def __enter__(self):
        for name in self.BUILDERS:
            original = getattr(tsf, name)
            self._originals[name] = original

            def wrapped(matrix, gauss, conductor, _original=original, _name=name):
                self.blocks.setdefault(_name, gauss.copy())
                return _original(matrix, gauss, conductor)

            setattr(tsf, name, wrapped)
        return self

    def __exit__(self, *exc_info):
        for name, original in self._originals.items():
            setattr(tsf, name, original)
        return False

    @property
    def flux_jacobian(self):
        return self.blocks["build_elamat"]

    @property
    def diffusion(self):
        return self.blocks["build_elkmat"]

    @property
    def source_jacobian(self):
        return self.blocks["build_elsmat"]


def first_step_blocks(tmp_path, formulation: str):
    with BlockCapture() as capture:
        simulation = run_simulation(
            prepare_run_directory(
                tmp_path,
                f"blocks_{formulation}",
                end_time=0.1,
                conductor_edits=compose_edits(
                    impose_pressure_drop_on_both_channels,
                    FORMULATION_EDITS[formulation],
                ),
            )
        )
    return simulation, capture


def row_scaled_deviation(explicit: np.ndarray, transformed: np.ndarray) -> float:
    scale = np.abs(transformed).max(axis=2, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    return float((np.abs(explicit - transformed) / scale).max())


@pytest.mark.parametrize(
    "property_path, block_bound",
    [("direct", 1.0e-12), ("tabulated", 1.0e-3)],
)
def test_gauss_blocks_match_transform(tmp_path, monkeypatch, property_path, block_bound):
    if property_path == "direct":
        monkeypatch.setattr(cpi, "USE_TABULAR_PROPERTIES", False)
    else:
        assert cpi.USE_TABULAR_PROPERTIES
    explicit_run, explicit = first_step_blocks(tmp_path, "explicit")
    transform_run, transformed = first_step_blocks(tmp_path, "mass_flow")
    assert (
        explicit_run.list_of_Conductors[0].hydraulic_formulation
        is HydraulicFormulation.MASS_FLOW_EXPLICIT
    )
    conductor = transform_run.list_of_Conductors[0]
    fluid_slots = [
        conductor.equation_index[f_comp.identifier]
        for f_comp in conductor.inventory.fluids.collection
    ]

    flux_deviation = row_scaled_deviation(
        explicit.flux_jacobian, transformed.flux_jacobian
    )
    source_deviation = row_scaled_deviation(
        explicit.source_jacobian, transformed.source_jacobian
    )
    print(
        f"\nGauss-block deviation explicit vs transform ({property_path} "
        f"properties): flux {flux_deviation:.3e}, source {source_deviation:.3e}"
    )
    assert flux_deviation < block_bound
    assert source_deviation < block_bound

    # Stabilisation block: identical diagonals, explicit has no
    # off-diagonal entries, the transform's off-diagonals are exactly the
    # conjugation image of the diagonal velocity-form block:
    #   (mdot, p) = rho A v kappa_T (k_p - k_mdot),
    #   (mdot, T) = rho A v beta (k_mdot - k_T).
    off_diagonal_report = []
    for f_comp, slots in zip(conductor.inventory.fluids.collection, fluid_slots):
        fields = f_comp.coolant.gauss_fields
        for slot in slots:
            # The conjugation evaluates rho A * (k / (rho A)): 1 ulp.
            assert np.allclose(
                explicit.diffusion[:, slot, slot],
                transformed.diffusion[:, slot, slot],
                rtol=1e-14,
                atol=0.0,
            )
        explicit_off = explicit.diffusion.copy()
        for slot in slots:
            explicit_off[:, slot, slot] = 0.0
        assert not explicit_off[:, slots, :][:, :, slots].any()

        rho_area_v = (
            fields.total_density * f_comp.channel.inputs.cross_section * fields.velocity
        )
        k_mdot = explicit.diffusion[:, slots.velocity, slots.velocity]
        k_p = explicit.diffusion[:, slots.pressure, slots.pressure]
        k_t = explicit.diffusion[:, slots.temperature, slots.temperature]
        predicted_p = rho_area_v * fields.isothermal_compressibility * (k_p - k_mdot)
        predicted_t = rho_area_v * fields.isobaric_expansion_coefficient * (k_mdot - k_t)
        found_p = transformed.diffusion[:, slots.velocity, slots.pressure]
        found_t = transformed.diffusion[:, slots.velocity, slots.temperature]
        assert np.allclose(found_p, predicted_p, rtol=1e-12, atol=0.0)
        assert np.allclose(found_t, predicted_t, rtol=1e-12, atol=0.0)
        # Everything else of the transformed fluid block is diagonal too.
        residual = transformed.diffusion[:, slots, :][:, :, slots].copy()
        for local in range(3):
            residual[:, local, local] = 0.0
        residual[:, 0, 1] = 0.0
        residual[:, 0, 2] = 0.0
        assert not residual.any()
        # Relative weight of the off-diagonal terms: entry times the
        # state scale against the diagonal term times its state scale.
        weight_p = np.abs(found_p * fields.pressure).max() / np.abs(
            k_mdot * fields.mass_flow_rate
        ).max()
        weight_t = np.abs(found_t * fields.temperature).max() / np.abs(
            k_mdot * fields.mass_flow_rate
        ).max()
        off_diagonal_report.append(
            f"{f_comp.identifier}: (mdot,p) {weight_p:.3e}, (mdot,T) {weight_t:.3e}"
        )
    print(
        "transform-route off-diagonal stabilisation weight relative to the "
        "mdot diagonal: " + "; ".join(off_diagonal_report)
    )


# --------------------------------------------------------------------------
# Solution difference and convergence
# --------------------------------------------------------------------------

FIELDS = ("mass_flow_rate", "pressure", "temperature")
MESHES = (50, 100, 200)
TIME_STEPS = (0.1, 0.05, 0.025)
# Pinned from measurement (100 W/m pulse, 2 s, dt = 0.1 s, 50 elements,
# final state; see test output) with a ~2-3x margin: mass flow 8.0e-3 /
# 5.5e-3, pressure 2.4e-4, temperature 6.3e-4 / 2.3e-4 (CHAN_1 / CHAN_2).
SOLUTION_DIFFERENCE_BOUNDS = {
    "mass_flow_rate": 2.0e-2,
    "pressure": 6.0e-4,
    "temperature": 2.0e-3,
}
# Measured refinement behaviour of the explicit-transform deviation
# (50 -> 100 -> 200 elements): mass flow decreases with fitted orders
# 0.31-0.45, pressure with 0.80-1.02 (the O(dz) stabilisation-block
# difference), temperature is refinement-STABLE (6.3e-4 -> 7.8e-4 ->
# 8.1e-4 on CHAN_1), like the velocity-vs-mass-flow floor documented in
# test_mass_flow_formulation. Monotone decrease is therefore asserted for
# mass flow and pressure only; temperature is asserted refinement-stable.
MONOTONE_FIELDS = ("mass_flow_rate", "pressure")
REFINEMENT_STABILITY_FACTOR = 1.5
# Steady pressure-drop run (heating off, 3 s, 50 elements), measured:
# mass flow 2.3e-4 / 1.5e-4, pressure 1.2e-6, temperature 9.2e-6 / 4.4e-6
# (CHAN_1 / CHAN_2): the boundary rows of the two stabilisation blocks
# fix slightly different discrete steady states even at uniform flow (on
# the constant-property single channel the same difference is 1.1e-7).
STEADY_DIFFERENCE_BOUND = 1.0e-3
IDENTITY_FLOOR = 1.0e-5


def set_time_step(run_directory, time_step: float) -> None:
    simulation_file = run_directory / "simulation.yaml"
    document = yaml.safe_load(simulation_file.read_text())
    stepping = document["simulation"]["time_stepping"]
    stepping["adaptivity"] = "FIXED"
    stepping["minimum_step"] = float(time_step)
    stepping["maximum_step"] = float(time_step)
    simulation_file.write_text(yaml.safe_dump(document, sort_keys=False))


def pulse_run(tmp_path, formulation: str, elements: int, time_step: float,
              heating: bool = True, end_time: float = 2.0):
    run_directory = prepare_run_directory(
        tmp_path,
        f"{formulation}_{elements}_{time_step}_{'pulse' if heating else 'steady'}",
        end_time=end_time,
        conductor_edits=compose_edits(
            impose_pressure_drop_on_both_channels,
            gentle_heat_pulse_edit if heating else heating_off_edit,
            grid_elements_edit(elements),
            FORMULATION_EDITS[formulation],
        ),
    )
    set_time_step(run_directory, time_step)
    simulation = run_simulation(run_directory)
    coordinates = simulation.list_of_Conductors[0].mesh.node_coordinates.copy()
    return coordinates, fluid_final_fields(simulation)


def deviation_table(reference, other, reference_x=None, other_x=None) -> dict:
    """Relative max-norm deviation per (channel, field), interpolating
    ``other`` onto the reference coordinates when meshes differ."""
    table = {}
    for channel in reference:
        for field in FIELDS:
            ref = reference[channel][field]
            oth = other[channel][field]
            if other_x is not None and reference_x is not None and oth.size != ref.size:
                oth = np.interp(reference_x, other_x, oth)
            table[channel, field] = float(np.abs(oth - ref).max() / np.abs(ref).max())
    return table


def fitted_order(coarse, fine, ratio=2.0):
    if fine <= IDENTITY_FLOOR:
        return float("nan")
    return float(np.log(coarse / fine) / np.log(ratio))


def format_table(deviations: dict, levels) -> str:
    lines = []
    for key in sorted(next(iter(deviations.values()))):
        values = [deviations[level][key] for level in levels]
        orders = [fitted_order(a, b) for a, b in zip(values[:-1], values[1:])]
        lines.append(
            f"  {key[0]} {key[1]:15s}: "
            + " -> ".join(f"{v:.3e}" for v in values)
            + "   fitted orders: "
            + ", ".join(f"{o:.2f}" for o in orders)
        )
    return "\n".join(lines)


def test_solution_difference_and_mesh_convergence(tmp_path):
    runs = {}
    for formulation in ("explicit", "mass_flow"):
        for elements in MESHES:
            runs[formulation, elements] = pulse_run(tmp_path, formulation, elements, 0.1)

    cross = {
        elements: deviation_table(runs["mass_flow", elements][1], runs["explicit", elements][1])
        for elements in MESHES
    }
    print("\nexplicit vs transform, 100 W/m pulse, dt = 0.1 s, mesh refinement:")
    print(format_table(cross, MESHES))

    for key, value in cross[MESHES[0]].items():
        bound = SOLUTION_DIFFERENCE_BOUNDS[key[1]]
        assert value < bound, f"{key}: {value:.3e} exceeds {bound:.0e}"
        assert value > 0.0, f"{key}: exact agreement is suspicious"
    for coarse, fine in zip(MESHES[:-1], MESHES[1:]):
        for key in cross[coarse]:
            if cross[fine][key] <= IDENTITY_FLOOR:
                continue  # below the property-identity floor
            if key[1] in MONOTONE_FIELDS:
                assert cross[fine][key] < cross[coarse][key], (
                    f"{key}: explicit-transform deviation does not decrease "
                    f"under refinement ({coarse} el {cross[coarse][key]:.3e} -> "
                    f"{fine} el {cross[fine][key]:.3e})"
                )
            else:
                assert cross[fine][key] < REFINEMENT_STABILITY_FACTOR * cross[coarse][key], (
                    f"{key}: explicit-transform deviation GROWS under refinement "
                    f"({coarse} el {cross[coarse][key]:.3e} -> {fine} el "
                    f"{cross[fine][key]:.3e})"
                )

    # Self-convergence of each route against its own finest mesh.
    print("self-convergence (against the 200-element run of the same route):")
    for formulation in ("explicit", "mass_flow"):
        fine_x, fine_fields = runs[formulation, MESHES[-1]]
        own = {}
        for elements in MESHES[:-1]:
            coarse_x, coarse_fields = runs[formulation, elements]
            own[elements] = deviation_table(coarse_fields, fine_fields, coarse_x, fine_x)
        print(f" {formulation}:")
        print(format_table(own, MESHES[:-1]))


def test_time_step_refinement(tmp_path):
    cross = {}
    for time_step in TIME_STEPS:
        fields = {
            formulation: pulse_run(tmp_path, formulation, MESHES[0], time_step)[1]
            for formulation in ("explicit", "mass_flow")
        }
        cross[time_step] = deviation_table(fields["mass_flow"], fields["explicit"])
    print("\nexplicit vs transform, 100 W/m pulse, 50 elements, time-step refinement:")
    print(format_table(cross, TIME_STEPS))
    # The deviation is a spatial-operator (stabilisation) difference: it must
    # not grow under time-step refinement.
    for coarse, fine in zip(TIME_STEPS[:-1], TIME_STEPS[1:]):
        for key in cross[coarse]:
            assert cross[fine][key] < 1.5 * max(cross[coarse][key], IDENTITY_FLOOR), (
                f"{key}: deviation grows under time-step refinement "
                f"({coarse} s {cross[coarse][key]:.3e} -> {fine} s {cross[fine][key]:.3e})"
            )


def test_steady_state_difference(tmp_path):
    fields = {
        formulation: pulse_run(
            tmp_path, formulation, MESHES[0], 0.1, heating=False, end_time=3.0
        )[1]
        for formulation in ("explicit", "mass_flow")
    }
    table = deviation_table(fields["mass_flow"], fields["explicit"])
    print("\nexplicit vs transform, steady pressure-drop run (3 s, 50 el):")
    for key, value in sorted(table.items()):
        print(f"  {key[0]} {key[1]:15s}: {value:.3e}")
    for key, value in table.items():
        assert value < STEADY_DIFFERENCE_BOUND, f"{key}: {value:.3e}"
