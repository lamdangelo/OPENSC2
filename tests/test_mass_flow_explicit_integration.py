"""Integration tests of the explicit (mdot, p, T) formulation
(``explicit_mass_flow_formulation: true``), run through the production
Simulation entry point and compared with the velocity formulation (and,
where cheap, with the similarity-transform mass-flow route):

* steady isothermal single-channel flow (constant-property fluid, laminar
  friction): the imposed pressure drop is reproduced by Darcy-Weisbach with
  the code's own friction factor, and the formulations agree to 1e-6;
* low-Mach pressure-pulse propagation: the front speed measured between
  two probes equals the fluid's speed of sound to 1 %;
* the CASE_1 two-channel deck (the multi-channel example used for the
  per-channel flow validation in tests/reference_data): per-channel inlet
  and outlet mass flows of every formulation against the committed
  reference values, printed as a table;
* mass-conservation diagnostic on the 100 W/m heated CASE_1 transient:
  the inventory integral rho A dz and the inlet-outlet mass-flow imbalance
  are logged per step for both formulations and the accumulated defect is
  REPORTED (not asserted - the primitive-variable scheme is not
  conservative, see test_mass_flow_formulation).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import interfaces.coolprop_interface as cpi
from hydraulics.hydraulic_flags import HydraulicFormulation
from simulation import Simulation

from regression_utilities import copy_input_files
from test_mass_flow_formulation import (
    compose_edits,
    formulation_edit,
    gentle_heat_pulse_edit,
)
from test_network_coupling import (
    CASE_1_INPUT_DIRECTORY,
    impose_pressure_drop_on_both_channels,
    prepare_run_directory,
    run_simulation,
)

sys.path.insert(0, str(Path(__file__).parent / "verification"))
from case_builders import (  # noqa: E402
    ACOUSTIC_FLUID,
    CHANNEL_CROSS_SECTION,
    CHANNEL_DIAMETER,
    CHANNEL_LENGTH,
    VERIFICATION_FLUID,
    WAVE_SPEED,
    channel_operations,
    run_case,
    write_channel_run_directory,
)

REFERENCE_DIRECTORY = (
    Path(__file__).parent / "reference_data" / "CASE_1_ITER_like_LTS"
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def explicit_flag_edit(conductor_document) -> None:
    conductor_document["conductor"]["inputs"][
        "explicit_mass_flow_formulation"
    ] = True


FORMULATION_EDITS = {
    "velocity": formulation_edit("velocity"),
    "mass_flow": formulation_edit("mass_flow"),
    "explicit": compose_edits(formulation_edit("mass_flow"), explicit_flag_edit),
}
RESOLVED = {
    "velocity": HydraulicFormulation.VELOCITY,
    "mass_flow": HydraulicFormulation.MASS_FLOW,
    "explicit": HydraulicFormulation.MASS_FLOW_EXPLICIT,
}


def set_channel_formulation(run_directory: Path, formulation: str) -> None:
    """Pin the formulation of a generated single-channel run directory."""
    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    document = yaml.safe_load(conductor_file.read_text())
    FORMULATION_EDITS[formulation](document)
    conductor_file.write_text(yaml.safe_dump(document, sort_keys=False))


def single_channel(simulation: Simulation):
    return simulation.list_of_Conductors[0].inventory.fluids.collection[0]


@pytest.fixture
def constant_fluid():
    previous = cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    yield VERIFICATION_FLUID
    cpi.set_constant_fluid_properties(previous)


@pytest.fixture
def acoustic_fluid():
    previous = cpi.set_constant_fluid_properties(ACOUSTIC_FLUID)
    yield ACOUSTIC_FLUID
    cpi.set_constant_fluid_properties(previous)


# --------------------------------------------------------------------------
# Steady isothermal flow: Darcy-Weisbach
# --------------------------------------------------------------------------

STEADY_PRESSURE_DROP = 200.0  # Pa (laminar: Re ~ 640 at the resulting flow)
STEADY_END_TIME = 20.0  # s (>> inertance/resistance time constant ~3 s;
# every number below is unchanged at 60 s: the runs are at their discrete
# fixed points)
CROSS_FORMULATION_TOLERANCE = 1.0e-6
# The three discrete steady states differ in the pressure PROFILE by up to
# 5.8e-5 of the imposed drop (5 mPa; 5e-8 of the absolute pressure), from
# the boundary rows of the different stabilisation blocks (the velocity
# form is additionally uniform in v, not mdot, see below); the mass flows
# agree to 8e-9. Pinned with margin from these measurements.
PRESSURE_PROFILE_TOLERANCE = 1.0e-4


def darcy_weisbach_mass_flow(channel, pressure_drop: float, density: float,
                             viscosity: float) -> float:
    """Fixed-point solution of 2 f(Re) rho v^2 L / D_h = dp with the
    channel's own friction model (total friction factor)."""
    mass_flow = 1.0e-3
    for _ in range(200):
        velocity = mass_flow / (density * CHANNEL_CROSS_SECTION)
        reynolds = abs(velocity) * density * CHANNEL_DIAMETER / viscosity
        friction = float(
            np.ravel(channel.channel.evaluate_friction_factors(
                np.array([reynolds])
            ).total)[0]
        )
        velocity_new = np.sqrt(
            pressure_drop * CHANNEL_DIAMETER / (2.0 * friction * density * CHANNEL_LENGTH)
        )
        mass_flow_new = density * CHANNEL_CROSS_SECTION * velocity_new
        if abs(mass_flow_new - mass_flow) < 1e-14 * mass_flow_new:
            return mass_flow_new
        mass_flow = mass_flow_new
    return mass_flow


def steady_run(tmp_path, formulation: str) -> Simulation:
    run_directory = write_channel_run_directory(
        tmp_path / f"steady_{formulation}",
        number_of_elements=40,
        time_step=0.1,
        end_time=STEADY_END_TIME,
        method="BDF2",
        hydraulic_boundary_condition=1,
        inlet_pressure=1.0e5 + STEADY_PRESSURE_DROP,
        outlet_pressure=1.0e5,
        initial_pressure=1.0e5 + 0.5 * STEADY_PRESSURE_DROP,
    )
    set_channel_formulation(run_directory, formulation)
    return run_case(run_directory)


def test_steady_darcy_weisbach_agreement(constant_fluid, tmp_path):
    runs = {name: steady_run(tmp_path, name) for name in FORMULATION_EDITS}
    for name, simulation in runs.items():
        assert simulation.list_of_Conductors[0].hydraulic_formulation is RESOLVED[name]

    fields = {
        name: single_channel(simulation).coolant.node_fields
        for name, simulation in runs.items()
    }
    mass_flows = {name: f.mass_flow_rate for name, f in fields.items()}
    reference = darcy_weisbach_mass_flow(
        single_channel(runs["velocity"]),
        STEADY_PRESSURE_DROP,
        constant_fluid.density,
        constant_fluid.viscosity,
    )
    report = "\n".join(
        f"  {name:10s}: mdot {mass_flows[name].mean():.9e} kg/s "
        f"(axial variation {np.ptp(mass_flows[name]) / abs(mass_flows[name].mean()):.2e}, "
        f"velocity variation {np.ptp(fields[name].velocity) / abs(fields[name].velocity.mean()):.2e}), "
        f"dp {fields[name].pressure[0] - fields[name].pressure[-1]:.6f} Pa, "
        f"DW deviation {abs(mass_flows[name].mean() - reference) / reference:.3e}"
        for name in runs
    )
    pairs = {}
    for name in ("mass_flow", "explicit"):
        for other in ("velocity", "mass_flow"):
            if other == name:
                continue
            pairs[name, other] = (
                abs(mass_flows[name].mean() - mass_flows[other].mean())
                / abs(mass_flows[other].mean()),
                np.abs(mass_flows[name] - mass_flows[other]).max()
                / abs(mass_flows[other].mean()),
                np.abs(fields[name].pressure - fields[other].pressure).max()
                / STEADY_PRESSURE_DROP,
            )
    pair_report = "\n".join(
        f"  {name} vs {other}: mean mdot {mean_dev:.3e}, nodal mdot {node_dev:.3e}, "
        f"pressure profile {p_dev:.3e} (relative to dp)"
        for (name, other), (mean_dev, node_dev, p_dev) in pairs.items()
    )
    print(
        f"\nsteady Darcy-Weisbach: mdot_DW = {reference:.9e} kg/s\n{report}\n"
        f"cross-formulation:\n{pair_report}"
    )

    for name in runs:
        # Pressure drop <-> Darcy-Weisbach with the code's own f. The
        # convective acceleration of the (slightly) compressible isothermal
        # flow, kappa_T dp ~ 2e-4, is the expected residual level.
        deviation = abs(mass_flows[name].mean() - reference) / reference
        assert deviation < 1.0e-3, f"{name}: mdot deviates from DW by {deviation:.3e}"
    # Steady continuity: the mass-flow routes are axially uniform in mdot.
    # The velocity formulation's discrete steady state is uniform in v
    # instead, so its rho*A*v carries the kappa_T*dp ~ 2e-4 density
    # variation (measured 2.0e-4; reported above, not asserted).
    for name in ("mass_flow", "explicit"):
        assert np.ptp(mass_flows[name]) / abs(mass_flows[name].mean()) < 1e-6, name

    for (name, other), (mean_dev, node_dev, p_dev) in pairs.items():
        assert mean_dev < CROSS_FORMULATION_TOLERANCE, (
            f"{name} vs {other}: mean mdot deviation {mean_dev:.3e}"
        )
        if other != "velocity":
            # Transform vs explicit: same unknowns, nodal agreement too.
            assert node_dev < CROSS_FORMULATION_TOLERANCE, (
                f"{name} vs {other}: nodal mdot deviation {node_dev:.3e}"
            )
        assert p_dev < PRESSURE_PROFILE_TOLERANCE, (
            f"{name} vs {other}: pressure deviation {p_dev:.3e} "
            "(relative to the pressure drop)"
        )
        assert p_dev * STEADY_PRESSURE_DROP / 1.0e5 < CROSS_FORMULATION_TOLERANCE, (
            f"{name} vs {other}: pressure deviation relative to the absolute "
            f"pressure {p_dev * STEADY_PRESSURE_DROP / 1.0e5:.3e}"
        )


# --------------------------------------------------------------------------
# Low-Mach pressure pulse: front speed
# --------------------------------------------------------------------------

PULSE_AMPLITUDE = 100.0  # Pa
PULSE_TIME = 0.05  # s
PULSE_STEADY_MASS_RATE = 5.0e-4  # kg/s (Mach 2e-4)
PULSE_ELEMENTS = 400
PULSE_TIME_STEP = 5.0e-4  # s (CFL = a dt / h = 0.63)
PROBE_POSITIONS = (2.5, 7.5)  # m


class InletPressureStep:
    def __init__(self, base_pressure, amplitude, step_time):
        self.base_pressure = base_pressure
        self.amplitude = amplitude
        self.step_time = step_time
        self.times = []
        self.pressures = []

    def __call__(self, simulation):
        channel = single_channel(simulation)
        self.times.append(simulation.simulation_time[-1])
        self.pressures.append(channel.coolant.node_fields.pressure.copy())
        if simulation.simulation_time[-1] >= self.step_time - 1e-12:
            channel_operations(simulation).inlet_pressure = (
                self.base_pressure + self.amplitude
            )
        return False


def crossing_time(times, signal, level):
    """First time the signal crosses the level (linear interpolation)."""
    above = np.nonzero(signal >= level)[0]
    assert above.size > 0, "front never reached the probe"
    index = above[0]
    assert index > 0
    fraction = (level - signal[index - 1]) / (signal[index] - signal[index - 1])
    return times[index - 1] + fraction * (times[index] - times[index - 1])


def pulse_run(tmp_path, formulation: str) -> tuple:
    run_directory = write_channel_run_directory(
        tmp_path / f"pulse_{formulation}",
        number_of_elements=PULSE_ELEMENTS,
        time_step=PULSE_TIME_STEP,
        end_time=PULSE_TIME + 0.9 * CHANNEL_LENGTH / WAVE_SPEED,
        method="BDF2",
        hydraulic_boundary_condition=2,
        inlet_pressure=1.0e5,
        outlet_pressure=1.0e5,
        initial_pressure=1.0e5,
        inlet_mass_rate=PULSE_STEADY_MASS_RATE,
        outlet_mass_rate=PULSE_STEADY_MASS_RATE,
    )
    set_channel_formulation(run_directory, formulation)
    driver = InletPressureStep(1.0e5, PULSE_AMPLITUDE, PULSE_TIME)
    simulation = run_case(run_directory, step_callback=driver)
    driver(simulation)  # final state
    coordinates = simulation.list_of_Conductors[0].mesh.node_coordinates
    return np.asarray(driver.times), np.array(driver.pressures), coordinates


def test_pressure_pulse_front_speed(acoustic_fluid, tmp_path):
    speeds = {}
    for formulation in ("velocity", "explicit"):
        times, pressures, coordinates = pulse_run(tmp_path, formulation)
        baseline = pressures[times < PULSE_TIME][-1]
        arrivals = []
        for position in PROBE_POSITIONS:
            node = int(np.argmin(np.abs(coordinates - position)))
            arrivals.append(
                crossing_time(
                    times, pressures[:, node] - baseline[node], 0.5 * PULSE_AMPLITUDE
                )
            )
        speeds[formulation] = (
            (PROBE_POSITIONS[1] - PROBE_POSITIONS[0]) / (arrivals[1] - arrivals[0])
        )
    report = ", ".join(
        f"{name}: {speed:.4f} m/s ({(speed / WAVE_SPEED - 1) * 100:+.3f} %)"
        for name, speed in speeds.items()
    )
    print(f"\npressure-pulse front speed (c = {WAVE_SPEED:.4f} m/s): {report}")
    for name, speed in speeds.items():
        assert abs(speed / WAVE_SPEED - 1.0) < 1.0e-2, f"{name}: {speed} vs {WAVE_SPEED}"


# --------------------------------------------------------------------------
# CASE_1 per-channel flow split against the committed reference
# --------------------------------------------------------------------------

FLOW_SPLIT_BOUND = 1.0e-2  # the reference itself is known to drift by 3e-4


def reference_flow_split() -> dict:
    values = {}
    for channel in ("CHAN_1", "CHAN_2"):
        frame = pd.read_csv(
            REFERENCE_DIRECTORY / "Time_evolution" / f"{channel}_inlet_outlet_te.tsv",
            sep="\t",
        )
        values[channel] = (
            float(frame["mass_flow_rate_inl (kg/s)"].iloc[-1]),
            float(frame["mass_flow_rate_out (kg/s)"].iloc[-1]),
        )
    return values


def full_case_1_run(tmp_path, formulation: str) -> Simulation:
    """The committed CASE_1 deck (100 s, 200 elements), formulation pinned."""
    run_directory = tmp_path / f"case1_{formulation}"
    run_directory.mkdir()
    copy_input_files(CASE_1_INPUT_DIRECTORY, run_directory, "yaml")
    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    document = yaml.safe_load(conductor_file.read_text())
    FORMULATION_EDITS[formulation](document)
    conductor_file.write_text(yaml.safe_dump(document, sort_keys=False))
    return run_simulation(run_directory)


def test_case_1_flow_split_table(tmp_path):
    """Per-channel inlet/outlet mass flows at TEND for every formulation
    against the committed CASE_1 reference (three full runs, ~1 min each)."""
    reference = reference_flow_split()
    rows = []
    results = {}
    for formulation in FORMULATION_EDITS:
        simulation = full_case_1_run(tmp_path, formulation)
        conductor = simulation.list_of_Conductors[0]
        assert conductor.hydraulic_formulation is RESOLVED[formulation]
        results[formulation] = {
            f_comp.identifier: (
                float(f_comp.coolant.node_fields.mass_flow_rate[0]),
                float(f_comp.coolant.node_fields.mass_flow_rate[-1]),
            )
            for f_comp in conductor.inventory.fluids.collection
        }
    header = f"{'channel':8s} {'quantity':10s} {'reference':>14s} " + " ".join(
        f"{name:>14s}" for name in FORMULATION_EDITS
    )
    rows.append(header)
    for channel in reference:
        for position, label in ((0, "mdot_inlet"), (1, "mdot_outlet")):
            row = f"{channel:8s} {label:10s} {reference[channel][position]:14.6e} "
            row += " ".join(
                f"{results[name][channel][position]:14.6e}" for name in FORMULATION_EDITS
            )
            rows.append(row)
    print("\nCASE_1 per-channel flow split at TEND [kg/s]:\n" + "\n".join(rows))
    for name in FORMULATION_EDITS:
        for channel in reference:
            for position in (0, 1):
                deviation = abs(
                    results[name][channel][position] - reference[channel][position]
                ) / abs(reference[channel][position])
                assert deviation < FLOW_SPLIT_BOUND, (
                    f"{name} {channel} position {position}: deviation "
                    f"{deviation:.3e} from reference"
                )


# --------------------------------------------------------------------------
# Mass-conservation diagnostic (heated transient)
# --------------------------------------------------------------------------


class MassInventoryLogger:
    """Per-step record of sum_channels integral rho A dz and of the net
    boundary inflow sum_channels (mdot_inlet - mdot_outlet)."""

    def __init__(self):
        self.times = []
        self.inventory = []
        self.net_inflow = []
        self.throughput = []

    def __call__(self, simulation):
        conductor = simulation.list_of_Conductors[0]
        coordinates = conductor.mesh.node_coordinates
        inventory = 0.0
        net_inflow = 0.0
        throughput = 0.0
        for f_comp in conductor.inventory.fluids.collection:
            fields = f_comp.coolant.node_fields
            area = f_comp.channel.inputs.cross_section
            inventory += float(np.trapezoid(fields.total_density * area, coordinates))
            net_inflow += float(fields.mass_flow_rate[0] - fields.mass_flow_rate[-1])
            throughput += abs(float(fields.mass_flow_rate[0]))
        self.times.append(simulation.simulation_time[-1])
        self.inventory.append(inventory)
        self.net_inflow.append(net_inflow)
        self.throughput.append(throughput)
        return False

    def frame(self) -> pd.DataFrame:
        times = np.asarray(self.times)
        inventory = np.asarray(self.inventory)
        net_inflow = np.asarray(self.net_inflow)
        steps = np.diff(times)
        # Right-endpoint quadrature: the backward-Euler update advances the
        # inventory with the new-time boundary flows.
        boundary_gain = np.concatenate(([0.0], np.cumsum(net_inflow[1:] * steps)))
        inventory_change = inventory - inventory[0]
        return pd.DataFrame(
            {
                "time (s)": times,
                "inventory (kg)": inventory,
                "net_inflow (kg/s)": net_inflow,
                "inventory_change (kg)": inventory_change,
                "boundary_gain (kg)": boundary_gain,
                "defect (kg)": inventory_change - boundary_gain,
                "throughput_integral (kg)": np.concatenate(
                    ([0.0], np.cumsum(np.asarray(self.throughput)[1:] * steps))
                ),
            }
        )


def test_mass_conservation_diagnostic(tmp_path):
    """Logs integral rho A dz and the inlet-outlet imbalance per step on the
    100 W/m heated CASE_1 transient for both formulations (and the
    transform route); reports the accumulated defect. Nothing is fixed
    or asserted beyond finiteness."""
    report = []
    for formulation in FORMULATION_EDITS:
        run_directory = prepare_run_directory(
            tmp_path,
            f"conservation_{formulation}",
            conductor_edits=compose_edits(
                impose_pressure_drop_on_both_channels,
                gentle_heat_pulse_edit,
                FORMULATION_EDITS[formulation],
            ),
        )
        logger = MassInventoryLogger()
        simulation = Simulation(str(run_directory), step_callback=logger)
        simulation.run()
        logger(simulation)  # final state
        frame = logger.frame()
        log_file = tmp_path / f"mass_conservation_{formulation}.tsv"
        frame.to_csv(log_file, sep="\t", index=False)
        assert np.isfinite(frame.to_numpy()).all()
        final = frame.iloc[-1]
        report.append(
            f"  {formulation:10s}: steps {len(frame) - 1}, inventory change "
            f"{final['inventory_change (kg)']:+.4e} kg, boundary gain "
            f"{final['boundary_gain (kg)']:+.4e} kg, accumulated defect "
            f"{final['defect (kg)']:+.4e} kg = "
            f"{abs(final['defect (kg)']) / final['throughput_integral (kg)']:.3e} "
            f"of the integrated inlet throughput (log: {log_file})"
        )
    print("\nmass-conservation diagnostic (100 W/m pulse, 2 s, 50 el):\n" + "\n".join(report))
