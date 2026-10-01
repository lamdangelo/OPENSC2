"""Tier definitions of the formulation suite: operating point, per-tier case
construction (deck + recorder + hooks) and the run function.

Levels: h, h2, h4 (and h8 for Tier 4) with dt proportional to dx unless a
tier variant fixes one of them (Tier 2 'dx' / 'dt' series).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from formulation_suite import geometry
from formulation_suite.geometry import ConductorSpec, NetworkCase
from formulation_suite.heat_pulse import GaussianPulse, PEAK_LINEAR_POWER, SIGMA_X_FRACTION
from formulation_suite.recorder import SuiteRecorder
from reference.mms_residual import (
    ManufacturedSolution, VelocityManufacturedSolution, residual as mms_residual,
)
from reference.steady_network import ChannelSpec, solve_steady_network
from reference.transmission_line import smooth_step

FORMULATIONS = ("velocity", "mass_flow")
LEVEL_FACTORS = {"h": 1, "h2": 2, "h4": 4, "h8": 8}
BASE_ELEMENTS = {"tier1": (100, 150, 200), "tier2": (200, 300, 400),
                 "tier3a": (200, 300, 400), "tier3b": (200, 300, 400)}
SETTLE_TIME_STEP = 2.0e-3  # s
# Thermal settling: the frictional/expansion temperature front advects at
# the flow speed (residence times 50-140 s), so the thermal steady state
# needs ~1500 s of simulated time; run it at a large implicit step (the
# discrete steady state does not depend on dt), then a short stage at the
# fine step before the tier phase.
SETTLE_COARSE = (0.5, 1500.0)  # (dt, duration)
SETTLE_FINE = (SETTLE_TIME_STEP, 2.0)
SETTLE_SCHEDULE = (SETTLE_COARSE, SETTLE_FINE)
SETTLE_TIME = sum(d for _, d in SETTLE_SCHEDULE)
TIER2_RAMP = 4.0e-3  # s: C-infinity step of the supply reservoir (0.9 m front)
TIER2_STEP = 1.0e3  # Pa (10 mbar)
TIER3A_BIAS = 1.0e-3  # Pa on the return reservoir (avoids the zero-flow NaN)
TIER4_LENGTH = 20.0
TIER4_ELEMENTS = 100
TIER4_TIME_STEP = 2.0e-3
TIER4_PERIOD = 0.5
TIER4_MASS_FLOW = 2.0e-3
TIER4_PRESSURE_DROP = 5.0e4
TIER4_SNAPSHOT_STRIDE = 1.0e-2  # s between stored state snapshots


# --------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------


@dataclass
class OperatingPoint:
    channel_specs: list
    inputs: object
    friction: object
    reference: object  # SteadyNetworkSolution with the volumes at 5.0/4.5 bar
    impedance: float  # Z_line = c/A at the mean state
    far_resistance: float
    supply_reservoir: float
    return_reservoir: float
    mach: dict = field(default_factory=dict)

    def mass_flows(self, specs) -> dict:
        return {spec.identifier: float(m) for spec, m in zip(specs, self.reference.mass_flows)}

    def to_json(self) -> dict:
        return {
            "impedance_Pa_s_per_kg": self.impedance,
            "far_resistance": self.far_resistance,
            "supply_reservoir": self.supply_reservoir,
            "return_reservoir": self.return_reservoir,
            "mass_flows": [float(m) for m in self.reference.mass_flows],
            "total_mass_flow": self.reference.total_mass_flow,
            "mach": self.mach,
            "pressure_supply_volume": self.reference.pressure_supply_volume,
            "pressure_return_volume": self.reference.pressure_return_volume,
        }


_OPERATING_POINT = None


def operating_point(supply=geometry.SUPPLY_PRESSURE, return_pressure=geometry.RETURN_PRESSURE,
                    cached: bool = True) -> OperatingPoint:
    """Tier-1 reference at the nominal volume pressures; the reservoir
    pressures are offset by R_far Q so that the coupled model's volumes sit
    exactly there in steady state."""
    global _OPERATING_POINT
    if cached and _OPERATING_POINT is not None and supply == geometry.SUPPLY_PRESSURE:
        return _OPERATING_POINT
    template = geometry.load_template()
    inputs = geometry.channel_inputs(template)
    friction = geometry.code_friction_law(inputs)
    specs = [ChannelSpec(f"CHAN1_C{k + 1}", L, inputs.cross_section, inputs.hydraulic_diameter, friction)
             for k, L in enumerate(geometry.LENGTHS)]
    reference = solve_steady_network(specs, supply, return_pressure, geometry.TEMPERATURE,
                                     0.0, 0.0, geometry.helium_properties, profile_points=401)
    mean_state = geometry.helium_properties(geometry.TEMPERATURE, 0.5 * (supply + return_pressure))
    impedance = mean_state["speed_of_sound"] / inputs.cross_section
    far_resistance = geometry.FAR_RESISTANCE_FACTOR * impedance
    total = reference.total_mass_flow
    mach = {}
    for spec, profile in zip(specs, reference.profiles):
        c = geometry.helium_properties(float(profile.temperature[0]), float(profile.pressure[0]))["speed_of_sound"]
        mach[spec.identifier] = float(np.abs(profile.velocity).max() / c)
    point = OperatingPoint(specs, inputs, friction, reference, impedance, far_resistance,
                           supply + far_resistance * total, return_pressure - far_resistance * total,
                           mach)
    if supply == geometry.SUPPLY_PRESSURE:
        _OPERATING_POINT = point
    return point


def conductor_specs(tier: str, factor: int) -> list:
    return [ConductorSpec(k + 1, L, int(n * factor))
            for k, (L, n) in enumerate(zip(geometry.LENGTHS, BASE_ELEMENTS[tier]))]


# --------------------------------------------------------------------------
# Run specification
# --------------------------------------------------------------------------


@dataclass
class RunSpec:
    tier: str
    formulation: str
    level: str
    variant: str
    run_directory: Path
    recorder: SuiteRecorder
    parameters: dict
    writer: object  # callable(run_directory) writing the deck
    post: object = None  # callable(simulation) after run (e.g. pulse calibration)

    @property
    def name(self) -> str:
        suffix = "" if self.variant == "prop" else f"_{self.variant}"
        return f"{self.formulation}_single_{self.level}{suffix}"


def _network_writer(case: NetworkCase):
    def write(run_directory):
        geometry.write_network_case(run_directory, case)
    return write


def tier1_spec(formulation: str, level: str, results_root: Path) -> RunSpec:
    factor = LEVEL_FACTORS[level]
    point = operating_point()
    specs = conductor_specs("tier1", factor)
    dt = SETTLE_TIME_STEP / factor
    tier_cap = 5.0
    case = NetworkCase(specs, SETTLE_COARSE[0], SETTLE_COARSE[1] + tier_cap, formulation,
                       supply_pressure=point.supply_reservoir, return_pressure=point.return_reservoir,
                       far_resistance=point.far_resistance,
                       pressure_supply_volume=point.reference.pressure_supply_volume,
                       pressure_return_volume=point.reference.pressure_return_volume,
                       mass_flows=point.mass_flows(specs),
                       spatial_times=[SETTLE_COARSE[1] + tier_cap], name="tier1")
    recorder = SuiteRecorder(settle_schedule=(SETTLE_COARSE,), tier_time_step=dt,
                             tier_duration=tier_cap, snapshot_times=[],
                             steady_tolerance=1.0e-12, settle_record_stride=1)
    params = {"tier": "tier1", "formulation": formulation, "level": level, "factor": factor,
              "time_step": dt, "elements": [s.elements for s in specs], "end_time_cap": tier_cap,
              "settle_schedule": [SETTLE_COARSE], "steady_tolerance": 1.0e-12,
              "operating_point": point.to_json()}
    directory = results_root / "tier1" / f"{formulation}_single_{level}"
    return RunSpec("tier1", formulation, level, "prop", directory, recorder, params,
                   _network_writer(case))


def tier2_spec(formulation: str, level: str, variant: str, results_root: Path) -> RunSpec:
    """variant: 'prop' (dt ~ dx), 'dx' (dt fixed at the h4 value), 'dt'
    (mesh fixed at h4, dt of the level)."""
    factor = LEVEL_FACTORS[level]
    point = operating_point()
    c_min = min(
        geometry.helium_properties(geometry.TEMPERATURE, p)["speed_of_sound"]
        for p in (point.reference.pressure_supply_volume, point.reference.pressure_return_volume)
    )
    mesh_factor = 4 if variant == "dt" else factor
    dt_factor = 4 if variant == "dx" else factor
    specs = conductor_specs("tier2", mesh_factor)
    dx_base = geometry.LENGTHS[0] / BASE_ELEMENTS["tier2"][0]
    dt = 0.2 * dx_base / c_min / dt_factor
    transit_1 = geometry.LENGTHS[0] / c_min
    duration = 6.0 * geometry.LENGTHS[2] / c_min
    snapshot_times = [0.5 * transit_1, 1.0 * transit_1, 2.0 * transit_1, 4.0 * transit_1, duration]
    case = NetworkCase(specs, SETTLE_COARSE[0], SETTLE_TIME + duration, formulation,
                       supply_pressure=point.supply_reservoir, return_pressure=point.return_reservoir,
                       far_resistance=point.far_resistance,
                       pressure_supply_volume=point.reference.pressure_supply_volume,
                       pressure_return_volume=point.reference.pressure_return_volume,
                       mass_flows=point.mass_flows(specs),
                       spatial_times=[SETTLE_TIME + duration], name="tier2")
    recorder = SuiteRecorder(settle_schedule=SETTLE_SCHEDULE,
                             tier_time_step=dt, tier_duration=duration,
                             snapshot_times=snapshot_times,
                             reservoir_schedule=smooth_step(TIER2_STEP, TIER2_RAMP),
                             settle_record_stride=10)
    params = {"tier": "tier2", "formulation": formulation, "level": level, "variant": variant,
              "mesh_factor": mesh_factor, "dt_factor": dt_factor, "time_step": dt,
              "elements": [s.elements for s in specs], "settle_time": SETTLE_TIME,
              "settle_schedule": list(SETTLE_SCHEDULE), "duration": duration,
              "snapshot_times": snapshot_times, "transit_time_channel_1": transit_1,
              "step_amplitude": TIER2_STEP, "step_ramp": TIER2_RAMP, "sound_speed_min": c_min,
              "operating_point": point.to_json()}
    suffix = "" if variant == "prop" else f"_{variant}"
    directory = results_root / "tier2" / f"{formulation}_single_{level}{suffix}"
    return RunSpec("tier2", formulation, level, variant, directory, recorder, params,
                   _network_writer(case))


def _pulse(length: float, centre_fraction: float, peak: float = PEAK_LINEAR_POWER) -> GaussianPulse:
    return GaussianPulse(centre_x=centre_fraction * length, sigma_x=SIGMA_X_FRACTION * length,
                         peak=peak)


def tier3a_spec(formulation: str, level: str, results_root: Path, peak: float | None = None,
                duration: float = 2.0, htc_model: int | None = None) -> RunSpec:
    """htc_model None: the W7-X deck's Dittus-Boelter-pure model (h -> 0 in
    stagnant flow); 1: Dittus-Boelter with the Nusselt lower limit (variant
    'htc1'), which removes the symmetry-breaking instability of the
    unfloored model at the expansion centre."""
    factor = LEVEL_FACTORS[level]
    specs = conductor_specs("tier3a", factor)
    for spec in specs:
        assert spec.elements % 2 == 0, "a node must sit at L/2: even element count"
    point = operating_point()
    dt = 1.0e-3 / factor
    supply = geometry.SUPPLY_PRESSURE
    case = NetworkCase(specs, SETTLE_TIME_STEP, 2.0 + duration, formulation,
                       supply_pressure=supply, return_pressure=supply - TIER3A_BIAS,
                       far_resistance=point.far_resistance,
                       pressure_supply_volume=supply, pressure_return_volume=supply - TIER3A_BIAS,
                       mass_flows={s.identifier: 1.0e-9 for s in specs},
                       heated_strands={specs[1].identifier: True},
                       spatial_times=[2.0 + duration], name="tier3a",
                       heat_transfer_model=htc_model)
    pulse = _pulse(geometry.LENGTHS[1], 0.5, peak or PEAK_LINEAR_POWER)
    recorder = SuiteRecorder(settle_schedule=((SETTLE_TIME_STEP, 2.0),),
                             tier_time_step=dt, tier_duration=duration,
                             snapshot_times=[0.05, 0.1, 0.2, 0.5, 1.0, duration],
                             heat_pulse=pulse, heated_conductor=specs[1].identifier,
                             store_full_mass_flow=True, settle_record_stride=10)
    params = {"tier": "tier3a", "formulation": formulation, "level": level, "factor": factor,
              "time_step": dt, "elements": [s.elements for s in specs], "settle_time": 2.0,
              "duration": duration, "bias_pressure": TIER3A_BIAS,
              "heat_transfer_model": htc_model or 2,
              "variant": "prop" if htc_model is None else f"htc{htc_model}",
              "pulse": {"centre_x": pulse.centre_x, "sigma_x": pulse.sigma_x,
                        "centre_t": pulse.centre_t, "sigma_t": pulse.sigma_t, "peak": pulse.peak,
                        "energy_J": pulse.total_energy()},
              "operating_point": point.to_json()}
    variant = "prop" if htc_model is None else f"htc{htc_model}"
    suffix = "" if variant == "prop" else f"_{variant}"
    directory = results_root / "tier3a" / f"{formulation}_single_{level}{suffix}"
    return RunSpec("tier3a", formulation, level, variant, directory, recorder, params,
                   _network_writer(case))


def tier3b_spec(formulation: str, level: str, results_root: Path,
                htc_model: int | None = None) -> RunSpec:
    factor = LEVEL_FACTORS[level]
    specs = conductor_specs("tier3b", factor)
    point = operating_point()
    dt = 1.0e-3 / factor
    duration = 2.0
    case = NetworkCase(specs, SETTLE_COARSE[0], SETTLE_TIME + duration, formulation,
                       supply_pressure=point.supply_reservoir, return_pressure=point.return_reservoir,
                       far_resistance=point.far_resistance,
                       pressure_supply_volume=point.reference.pressure_supply_volume,
                       pressure_return_volume=point.reference.pressure_return_volume,
                       mass_flows=point.mass_flows(specs),
                       heated_strands={specs[1].identifier: True},
                       spatial_times=[SETTLE_TIME + duration], name="tier3b",
                       heat_transfer_model=htc_model)
    pulse = _pulse(geometry.LENGTHS[1], 1.0 / 3.0)
    recorder = SuiteRecorder(settle_schedule=SETTLE_SCHEDULE,
                             tier_time_step=dt, tier_duration=duration,
                             snapshot_times=[0.1, 0.2, 0.5, 1.0, duration],
                             heat_pulse=pulse, heated_conductor=specs[1].identifier,
                             settle_record_stride=10)
    params = {"tier": "tier3b", "formulation": formulation, "level": level, "factor": factor,
              "time_step": dt, "elements": [s.elements for s in specs], "settle_time": SETTLE_TIME,
              "settle_schedule": list(SETTLE_SCHEDULE), "duration": duration,
              "heat_transfer_model": htc_model or 2,
              "variant": "prop" if htc_model is None else f"htc{htc_model}",
              "pulse": {"centre_x": pulse.centre_x, "sigma_x": pulse.sigma_x,
                        "centre_t": pulse.centre_t, "sigma_t": pulse.sigma_t, "peak": pulse.peak,
                        "energy_J": pulse.total_energy()},
              "operating_point": point.to_json()}
    variant = "prop" if htc_model is None else f"htc{htc_model}"
    suffix = "" if variant == "prop" else f"_{variant}"
    directory = results_root / "tier3b" / f"{formulation}_single_{level}{suffix}"
    return RunSpec("tier3b", formulation, level, variant, directory, recorder, params,
                   _network_writer(case))


# --------------------------------------------------------------------------
# Tier 4: manufactured solution
# --------------------------------------------------------------------------


def manufactured_solution() -> ManufacturedSolution:
    return ManufacturedSolution(mass_flow_0=TIER4_MASS_FLOW, pressure_0=geometry.SUPPLY_PRESSURE,
                                pressure_drop=TIER4_PRESSURE_DROP, length=TIER4_LENGTH,
                                period=TIER4_PERIOD, temperature_0=geometry.TEMPERATURE)


class ManufacturedSource:
    """fluid_source_callback of the Tier-4 conductor: residual rows of the
    formulation's PDE at the nodes, at the new time level (and at the previous
    level for the two-column first step)."""

    def __init__(self, solution: ManufacturedSolution, formulation: str, inputs):
        self.solution = solution
        self.formulation = formulation
        self.area = inputs.cross_section
        self.hydraulic_diameter = inputs.hydraulic_diameter
        self.friction = geometry.code_friction_law(inputs)

    def rows(self, x, t):
        return mms_residual(self.solution, x, t, self.formulation, geometry.helium_properties,
                            self.friction, self.area, self.hydraulic_diameter)

    def _fill(self, array, conductor, t):
        x = conductor.mesh.node_coordinates
        rows = self.rows(x, t)
        for f_comp in conductor.inventory.fluids.collection:
            eq_idx = conductor.equation_index[f_comp.identifier]
            for slot, key in ((eq_idx.velocity, "velocity"), (eq_idx.pressure, "pressure"),
                              (eq_idx.temperature, "temperature")):
                array[:, slot, 0] += rows[key][:-1]
                array[:, slot, 1] += rows[key][1:]

    def __call__(self, conductor, source_vector):
        t_new = float(conductor.cond_time[-1])
        if hasattr(source_vector, "present"):
            self._fill(source_vector.present, conductor, t_new)
            self._fill(source_vector.previous, conductor, float(conductor.cond_time[-2]))
        else:
            self._fill(source_vector, conductor, t_new)
        return source_vector


def manufactured_initial_state(solution: ManufacturedSolution):
    """Overwrite the solver's own initial channel state with W_M(x, 0)."""

    def impose(simulation):
        for conductor in simulation.list_of_Conductors:
            x = conductor.mesh.node_coordinates
            ndf = conductor.equation_counts.degrees_of_freedom_per_node
            for f_comp in conductor.inventory.fluids.collection:
                fields = f_comp.coolant.node_fields
                eq_idx = conductor.equation_index[f_comp.identifier]
                pressure = solution.pressure(x, 0.0)
                temperature = solution.temperature(x, 0.0)
                mass_flow = solution.mass_flow(x, 0.0)
                density = geometry.helium_properties(temperature, pressure)["density"]
                fields.pressure = pressure
                fields.temperature = temperature
                fields.total_density = density
                fields.mass_flow_rate = mass_flow
                fields.velocity = mass_flow / (density * f_comp.channel.inputs.cross_section)
                sol = conductor.time_integration.solution
                from hydraulics.hydraulic_flags import MASS_FLOW_FORMULATIONS
                slot_value = (mass_flow if conductor.hydraulic_formulation in MASS_FLOW_FORMULATIONS
                              else fields.velocity)
                for column in range(sol.shape[1]):
                    sol[eq_idx.velocity::ndf, column] = slot_value
                    sol[eq_idx.pressure::ndf, column] = pressure
                    sol[eq_idx.temperature::ndf, column] = temperature
    return impose


def manufactured_solution_velocity() -> VelocityManufacturedSolution:
    """Tier-5 solution: the velocity is the prescribed sinusoid and the mass
    flow rho A v the derived field (same p_M, T_M, base state and boundary
    values as Tier 4)."""
    inputs = geometry.channel_inputs(geometry.load_template())
    return VelocityManufacturedSolution(
        mass_flow_0=TIER4_MASS_FLOW, pressure_0=geometry.SUPPLY_PRESSURE,
        pressure_drop=TIER4_PRESSURE_DROP, length=TIER4_LENGTH, period=TIER4_PERIOD,
        temperature_0=geometry.TEMPERATURE, properties=geometry.helium_properties,
        area=inputs.cross_section)


def tier4_spec(formulation: str, level: str, results_root: Path, tier: str = "tier4") -> RunSpec:
    """Manufactured-solution run. ``tier`` selects the solution: 'tier4'
    prescribes the mass flow (velocity derived), 'tier5' prescribes the
    velocity (mass flow derived); mesh, step, boundary values and recorder
    are identical."""
    factor = LEVEL_FACTORS[level]
    solution = manufactured_solution() if tier == "tier4" else manufactured_solution_velocity()
    inputs = geometry.channel_inputs(geometry.load_template())
    elements = TIER4_ELEMENTS * factor
    dt = TIER4_TIME_STEP / factor
    directory = results_root / tier / f"{formulation}_single_{level}"

    def write(run_directory):
        geometry.write_single_channel_case(
            run_directory, TIER4_LENGTH, elements, dt, TIER4_PERIOD, formulation,
            inlet_mass_flow=solution.mass_flow_0, inlet_pressure=solution.pressure_0,
            outlet_pressure=solution.pressure_0 - solution.pressure_drop,
            temperature=solution.temperature_0, spatial_times=[TIER4_PERIOD],
        )

    # Dense snapshots (every 10 ms, plus the baseline at t = 0) for the error
    # history; the quarter-period instants are on this grid.
    snapshot_times = list(np.arange(TIER4_SNAPSHOT_STRIDE, TIER4_PERIOD + 1e-9, TIER4_SNAPSHOT_STRIDE))
    recorder = SuiteRecorder(tier_time_step=dt, tier_duration=TIER4_PERIOD,
                             snapshot_times=snapshot_times,
                             fluid_source=ManufacturedSource(solution, formulation, inputs),
                             initial_state=manufactured_initial_state(solution))
    params = {"tier": tier, "formulation": formulation, "level": level, "factor": factor,
              "time_step": dt, "elements": [elements], "length": TIER4_LENGTH,
              "period": TIER4_PERIOD, "manufactured": {
                  "prescribed_flow_variable": "mass_flow" if tier == "tier4" else "velocity",
                  "mass_flow_0": solution.mass_flow_0, "pressure_0": solution.pressure_0,
                  "pressure_drop": solution.pressure_drop, "temperature_0": solution.temperature_0}}
    return RunSpec(tier, formulation, level, "prop", directory, recorder, params, write)


# --------------------------------------------------------------------------
# Matrix
# --------------------------------------------------------------------------


def run_specs(tier: str, formulations, levels, results_root: Path, quick: bool = False) -> list:
    specs = []
    levels = list(levels)
    if quick:
        levels = ["h"]
    for formulation in formulations:
        for level in levels:
            if tier == "tier1":
                specs.append(tier1_spec(formulation, level, results_root))
            elif tier == "tier2":
                variants = ["prop"] if quick else ["prop"]
                specs.append(tier2_spec(formulation, level, "prop", results_root))
                if not quick and level in ("h", "h2"):
                    specs.append(tier2_spec(formulation, level, "dx", results_root))
                    specs.append(tier2_spec(formulation, level, "dt", results_root))
            elif tier == "tier3a":
                specs.append(tier3a_spec(formulation, level, results_root))
                specs.append(tier3a_spec(formulation, level, results_root, htc_model=1))
            elif tier == "tier3b":
                specs.append(tier3b_spec(formulation, level, results_root))
                specs.append(tier3b_spec(formulation, level, results_root, htc_model=1))
            elif tier in ("tier4", "tier5"):
                specs.append(tier4_spec(formulation, level, results_root, tier=tier))
                if not quick and level == "h4":
                    specs.append(tier4_spec(formulation, "h8", results_root, tier=tier))
            else:
                raise ValueError(tier)
    return specs


def save_operating_point(results_root: Path) -> None:
    point = operating_point()
    reference = point.reference
    payload = point.to_json()
    payload["profiles"] = {
        spec.identifier: {
            "x": profile.x.tolist(), "pressure": profile.pressure.tolist(),
            "temperature": profile.temperature.tolist(), "density": profile.density.tolist(),
            "velocity": profile.velocity.tolist(),
        }
        for spec, profile in zip(point.channel_specs, reference.profiles)
    }
    payload["conductor_template"] = str(geometry.TEMPLATE_DECK.relative_to(geometry.REPOSITORY_ROOT))
    payload["channel"] = {
        "cross_section": point.inputs.cross_section,
        "hydraulic_diameter": point.inputs.hydraulic_diameter,
        "void_fraction": point.inputs.void_fraction,
        "friction_factor_model": point.inputs.friction_factor_model.name,
        "friction_multiplier": point.inputs.friction_multiplier,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "operating_point.json").write_text(json.dumps(payload, indent=1))
