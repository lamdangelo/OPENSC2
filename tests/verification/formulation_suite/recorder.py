"""Step-callback recorder of the formulation suite.

One object drives a whole run through ``Simulation(step_callback=...)``:

* phase control: an optional settling phase at a coarse fixed time step
  (constant boundary conditions), then the tier phase at the tier's time
  step with the tier's excitation (reservoir pressure schedule and/or heat
  pulse), both switched through ``simulation.transient_input["STPMIN"]``
  (the FIXED-step controller re-reads it every step) - one run, no restart;
* hooks installed at the first call (heat pulse user function, Tier-4 fluid
  source callback, Tier-4 initial state);
* per-step records: channel fields at the nodes (snapshots at requested
  times), port flows AS PASSED TO THE NETWORK, node masses M_i = V rho(p_i,T_i),
  the port balance residual r_port,i and the interior mass defect d_int,k,
  the steady-state change ||dW||/||W||;
* writers for state_snapshots.npz, port_timeseries.csv, diagnostics.csv,
  run_manifest.json.

Port flow as passed to the network (finding, see summary.md): in the
velocity formulation the Schur block and the node enthalpy balance use
sigma * rho^n * A * v^{n+1} (old-step density at the port node), in the
mass-flow formulations sigma * mdot^{n+1}. The recorder reproduces this with
the port-node density it stored at the previous call.

Time convention: the callback fires at the top of every iteration with the
previous step complete; records are stamped with the completed time. The
time step that produced record n is t_n - t_{n-1}.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import CoolProp
from hydraulics.hydraulic_flags import MASS_FLOW_FORMULATIONS

from formulation_suite.geometry import REPOSITORY_ROOT, helium_properties

SUPPLY_NODE = "supply_volume"
RETURN_NODE = "return_volume"
SUPPLY_RESERVOIR = "supply_bc"
SUPPLY_VALVE = "supply_valve"
RETURN_VALVE = "return_valve"


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def property_library_version() -> dict:
    return {
        "CoolProp": getattr(CoolProp, "__version__", "unknown"),
        "CoolProp_git": getattr(CoolProp, "__gitrevision__", "unknown"),
        "tabulated_properties": bool(
            __import__("interfaces.coolprop_interface", fromlist=["x"]).USE_TABULAR_PROPERTIES
        ),
    }


@dataclass
class ChannelRecord:
    identifier: str
    conductor: str
    x: np.ndarray
    area: float
    snapshot_times: list = field(default_factory=list)
    mass_flow: list = field(default_factory=list)
    pressure: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    density: list = field(default_factory=list)
    strand_temperature: list = field(default_factory=list)
    full_mass_flow: list = field(default_factory=list)  # every step (optional)
    full_times: list = field(default_factory=list)


class SuiteRecorder:
    def __init__(self, *, settle_schedule=(), tier_time_step: float, tier_duration: float,
                 snapshot_times=(), reservoir_schedule=None, heat_pulse=None,
                 heated_conductor: str | None = None, fluid_source=None, initial_state=None,
                 steady_tolerance: float | None = None, store_full_mass_flow: bool = False,
                 settle_record_stride: int = 1):
        """settle_schedule: sequence of (time_step, duration) stages run with
        constant boundary conditions before the tier phase (e.g. a long
        stage at a large implicit step to reach the thermal steady state,
        then a short stage at the tier step)."""
        self.settle_schedule = [(float(dt), float(duration)) for dt, duration in settle_schedule]
        self.settle_time = float(sum(duration for _, duration in self.settle_schedule))
        self.tier_time_step = tier_time_step
        self.tier_duration = tier_duration
        self.snapshot_times = sorted(float(t) for t in snapshot_times)
        self.reservoir_schedule = reservoir_schedule
        self.heat_pulse = heat_pulse
        self.heated_conductor = heated_conductor
        self.fluid_source = fluid_source
        self.initial_state = initial_state
        self.steady_tolerance = steady_tolerance
        self.store_full_mass_flow = store_full_mass_flow
        self.settle_record_stride = settle_record_stride

        self.started = False
        self.active = False
        self.time_origin = 0.0
        self.channels: dict = {}
        self.ports: list = []
        self.formulation = None
        self.supply_pressure_0 = None
        self.previous = None  # dict of previous-step quantities
        self.rows_ports: list = []
        self.rows_diagnostics: list = []
        self.pending_snapshots = list(self.snapshot_times)
        self.steady_change_last = np.nan
        self.stopped_steady = False
        self.settle_steps = 0
        self.tier_steps = 0
        self.max_strand_temperature = {}  # conductor -> (T, t_rel, x)
        self.wall_start = time.perf_counter()

    # ------------------------------------------------------------------ setup
    def _setup(self, simulation):
        network = simulation.hydraulic_network
        for conductor in simulation.list_of_Conductors:
            self.formulation = conductor.hydraulic_formulation
            for f_comp in conductor.inventory.fluids.collection:
                self.channels[f_comp.identifier] = ChannelRecord(
                    f_comp.identifier, conductor.identifier,
                    conductor.mesh.node_coordinates.copy(),
                    float(f_comp.channel.inputs.cross_section),
                )
            for port in conductor.network_ports:
                self.ports.append(port)
            if self.heat_pulse is not None and conductor.identifier == self.heated_conductor:
                strand = next(s for s in conductor.inventory.solids.collection
                              if s.identifier.startswith("STR_MIX"))
                self.heat_pulse.install(strand)
            if self.fluid_source is not None:
                conductor.fluid_source_callback = self.fluid_source
        if network is not None:
            self.supply_pressure_0 = float(network.node_pressure_of(SUPPLY_RESERVOIR))
        if self.initial_state is not None:
            self.initial_state(simulation)
        self.stage_ends = np.cumsum([duration for _, duration in self.settle_schedule])
        if self.settle_time > 0.0:
            self._set_time_step(simulation, self.settle_schedule[0][0])
        else:
            self._activate(simulation, 0.0)
        self.started = True

    @staticmethod
    def _set_time_step(simulation, time_step):
        simulation.transient_input["STPMIN"] = float(time_step)
        simulation.transient_input["STPMAX"] = float(time_step)

    def _settle_stage_time_step(self, t):
        for (time_step, _), end in zip(self.settle_schedule, self.stage_ends):
            if t < end - 1e-9:
                return time_step
        return self.settle_schedule[-1][0]

    def _activate(self, simulation, t):
        self.active = True
        self.time_origin = t
        self._set_time_step(simulation, self.tier_time_step)
        if self.heat_pulse is not None:
            self.heat_pulse.time_origin = t
            self.heat_pulse.active = True
        # Baseline snapshot: the settled state at t_rel = 0.
        self._snapshot(simulation, t)

    # ------------------------------------------------------------------ call
    def __call__(self, simulation) -> bool:
        t = float(simulation.simulation_time[-1])
        if not self.started:
            self._setup(simulation)
            self._record(simulation, t, first=True)
            return False
        if not self.active and t >= self.settle_time - 1e-9:
            self._activate(simulation, t)
        elif not self.active:
            self._set_time_step(simulation, self._settle_stage_time_step(t))
        if self.active:
            self.tier_steps += 1
        else:
            self.settle_steps += 1
        record_now = self.active or (self.settle_steps % self.settle_record_stride == 0)
        self._record(simulation, t, first=False, store=record_now)
        # Boundary schedule for the upcoming step (evaluate at t_{n+1}).
        if self.active and self.reservoir_schedule is not None:
            t_next = t + float(simulation.transient_input["STPMIN"])
            simulation.hydraulic_network.set_reservoir_state(
                SUPPLY_RESERVOIR,
                pressure=self.supply_pressure_0 + float(self.reservoir_schedule(t_next - self.time_origin)),
            )
        # Steady-state stop (Tier 1): measured at the tier time step.
        if self.steady_tolerance is not None and self.active and self.tier_steps >= 3:
            if self.steady_change_last < self.steady_tolerance:
                self.stopped_steady = True
                self._snapshot(simulation, t)
                return True
        return False

    # ------------------------------------------------------------------ record
    def _steady_change(self, simulation) -> float:
        worst = 0.0
        for conductor in simulation.list_of_Conductors:
            solution = conductor.time_integration.solution
            new, old = solution[:, 0], solution[:, 1]
            norm = np.linalg.norm(new)
            if norm > 0.0:
                worst = max(worst, float(np.linalg.norm(new - old) / norm))
        return worst

    def _port_flow_as_passed(self, port, previous_density) -> float:
        fields = port.fluid_component.coolant.node_fields
        end = port.end_node_slice
        if self.formulation in MASS_FLOW_FORMULATIONS:
            return float(port.flow_orientation_sign * fields.mass_flow_rate[end])
        if previous_density is None:
            return np.nan
        return float(
            port.flow_orientation_sign * previous_density
            * port.fluid_component.channel.inputs.cross_section * fields.velocity[end]
        )

    def _record(self, simulation, t, first: bool, store: bool = True):
        network = simulation.hydraulic_network
        t_rel = t - self.time_origin if self.active else t - self.settle_time
        self.steady_change_last = self._steady_change(simulation) if not first else np.nan
        # Node masses.
        node_mass = {}
        node_state = {}
        if network is not None:
            for node in (SUPPLY_NODE, RETURN_NODE):
                p = float(network.node_pressure_of(node))
                T = float(network.node_temperature_of(node))
                index = network.node_index[node] if hasattr(network, "node_index") else None
                volume = float(network.node_volume[index]) if index is not None else np.nan
                node_mass[node] = volume * helium_properties(T, p)["density"]
                node_state[node] = (p, T)
        # Port flows as passed and channel-end flows.
        port_flows = {}
        port_density_now = {}
        for port in self.ports:
            key = (port.fluid_component.identifier, "inlet" if port.end_node_slice == 0 else "outlet")
            previous_density = None if self.previous is None else self.previous["port_density"].get(key)
            port_flows[key] = self._port_flow_as_passed(port, previous_density)
            port_density_now[key] = float(
                port.fluid_component.coolant.node_fields.total_density[port.end_node_slice]
            )
        # Channel inventories and end flows.
        inventory = {}
        end_flows = {}
        for conductor in simulation.list_of_Conductors:
            x = conductor.mesh.node_coordinates
            for f_comp in conductor.inventory.fluids.collection:
                fields = f_comp.coolant.node_fields
                area = f_comp.channel.inputs.cross_section
                density = helium_properties(fields.temperature, fields.pressure)["density"]
                inventory[f_comp.identifier] = float(np.trapezoid(density * area, x))
                end_flows[f_comp.identifier] = (
                    float(fields.mass_flow_rate[0]), float(fields.mass_flow_rate[-1])
                )
        # Diagnostics relative to the previous record.
        row_d = {"time": t, "time_relative": t_rel, "phase": "tier" if self.active else "settle",
                 "picard_iters": 1, "steady_change": self.steady_change_last}
        row_p = {"time": t, "time_relative": t_rel}
        if self.previous is not None:
            dt = t - self.previous["time"]
            if dt > 0.0 and network is not None:
                valve_in = float(network.branch_mass_flow_of(SUPPLY_VALVE))
                valve_out = float(network.branch_mass_flow_of(RETURN_VALVE))
                inflow = {SUPPLY_NODE: valve_in, RETURN_NODE: -valve_out}
                for (channel, end), flow in port_flows.items():
                    node = SUPPLY_NODE if end == "inlet" else RETURN_NODE
                    inflow[node] += flow  # flow already signed "into node"
                for node in (SUPPLY_NODE, RETURN_NODE):
                    row_d[f"r_port_{node}"] = (
                        (node_mass[node] - self.previous["node_mass"][node]) / dt - inflow[node]
                    )
                    row_d[f"net_inflow_{node}"] = inflow[node]
            if dt > 0.0:
                for channel, mass in inventory.items():
                    m_in, m_out = end_flows[channel]
                    row_d[f"d_int_{channel}"] = (
                        mass - self.previous["inventory"][channel] - dt * (m_in - m_out)
                    )
            row_d["time_step"] = dt
        for node in node_mass:
            row_p[f"M_{node}"] = node_mass[node]
            row_p[f"p_{node}"] = node_state[node][0]
            row_p[f"T_{node}"] = node_state[node][1]
        if network is not None:
            row_p["mdot_supply_valve"] = float(network.branch_mass_flow_of(SUPPLY_VALVE))
            row_p["mdot_return_valve"] = float(network.branch_mass_flow_of(RETURN_VALVE))
            row_p["p_supply_bc"] = float(network.node_pressure_of(SUPPLY_RESERVOIR))
        for (channel, end), flow in port_flows.items():
            row_p[f"mdot_port_{end}_{channel}"] = flow
        for channel, (m_in, m_out) in end_flows.items():
            row_p[f"mdot_channel_in_{channel}"] = m_in
            row_p[f"mdot_channel_out_{channel}"] = m_out
        for channel, record in self.channels.items():
            f_comp = self._fluid(simulation, channel)
            fields = f_comp.coolant.node_fields
            mid = np.argmin(np.abs(record.x - 0.5 * record.x[-1]))
            row_p[f"mdot_mid_{channel}"] = float(fields.mass_flow_rate[mid])
            strand = self._strand(simulation, record.conductor)
            if strand is not None:
                T_s = strand.node_fields.temperature
                i_max = int(np.argmax(T_s))
                row_p[f"Tmax_strand_{record.conductor}"] = float(T_s[i_max])
                current = self.max_strand_temperature.get(record.conductor, (-np.inf, 0.0, 0.0))
                if T_s[i_max] > current[0]:
                    self.max_strand_temperature[record.conductor] = (
                        float(T_s[i_max]), t_rel, float(record.x[i_max])
                    )
            row_p[f"Tmax_fluid_{channel}"] = float(fields.temperature.max())
            if self.store_full_mass_flow and self.active:
                record.full_mass_flow.append(fields.mass_flow_rate.copy())
                record.full_times.append(t_rel)
        if store:
            self.rows_ports.append(row_p)
            self.rows_diagnostics.append(row_d)
        self.previous = {"time": t, "node_mass": node_mass, "inventory": inventory,
                         "port_density": port_density_now}
        # Snapshots at requested relative times (nearest completed step).
        if self.active and self.pending_snapshots:
            dt_tier = float(simulation.transient_input["STPMIN"])
            while self.pending_snapshots and t_rel >= self.pending_snapshots[0] - 0.5 * dt_tier:
                self.pending_snapshots.pop(0)
                self._snapshot(simulation, t)

    def _fluid(self, simulation, channel):
        for conductor in simulation.list_of_Conductors:
            for f_comp in conductor.inventory.fluids.collection:
                if f_comp.identifier == channel:
                    return f_comp
        raise KeyError(channel)

    @staticmethod
    def _strand(simulation, conductor_id):
        for conductor in simulation.list_of_Conductors:
            if conductor.identifier == conductor_id:
                for s_comp in conductor.inventory.solids.collection:
                    if s_comp.identifier.startswith("STR_MIX"):
                        return s_comp
        return None

    def _snapshot(self, simulation, t):
        t_rel = t - self.time_origin
        for channel, record in self.channels.items():
            f_comp = self._fluid(simulation, channel)
            fields = f_comp.coolant.node_fields
            record.snapshot_times.append(t_rel)
            record.mass_flow.append(fields.mass_flow_rate.copy())
            record.pressure.append(fields.pressure.copy())
            record.temperature.append(fields.temperature.copy())
            record.velocity.append(fields.velocity.copy())
            record.density.append(helium_properties(fields.temperature, fields.pressure)["density"])
            strand = self._strand(simulation, record.conductor)
            record.strand_temperature.append(
                strand.node_fields.temperature.copy() if strand is not None else np.full_like(record.x, np.nan)
            )

    def finish(self, simulation):
        """Record the final state (the callback does not see it)."""
        t = float(simulation.simulation_time[-1])
        if not self.stopped_steady:
            self._record(simulation, t, first=False)
            if self.pending_snapshots or self.steady_tolerance is not None:
                self._snapshot(simulation, t)
        self.wall_time = time.perf_counter() - self.wall_start

    # ------------------------------------------------------------------ write
    def write(self, directory: Path, manifest: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for channel, record in self.channels.items():
            arrays[f"x_{channel}"] = record.x
            arrays[f"t_snap_{channel}"] = np.asarray(record.snapshot_times)
            arrays[f"mdot_{channel}"] = np.asarray(record.mass_flow)
            arrays[f"p_{channel}"] = np.asarray(record.pressure)
            arrays[f"T_{channel}"] = np.asarray(record.temperature)
            arrays[f"v_{channel}"] = np.asarray(record.velocity)
            arrays[f"rho_{channel}"] = np.asarray(record.density)
            arrays[f"Tstrand_{channel}"] = np.asarray(record.strand_temperature)
            arrays[f"area_{channel}"] = np.asarray(record.area)
            if record.full_mass_flow:
                arrays[f"mdot_full_{channel}"] = np.asarray(record.full_mass_flow)
                arrays[f"t_full_{channel}"] = np.asarray(record.full_times)
        np.savez_compressed(directory / "state_snapshots.npz", **arrays)
        pd.DataFrame(self.rows_ports).to_csv(directory / "port_timeseries.csv", index=False)
        pd.DataFrame(self.rows_diagnostics).to_csv(directory / "diagnostics.csv", index=False)
        manifest = dict(manifest)
        manifest.update(
            {
                "git_hash": git_hash(),
                "property_library": property_library_version(),
                "wall_time_s": getattr(self, "wall_time", np.nan),
                "settle_steps": self.settle_steps,
                "settle_schedule": self.settle_schedule,
                "tier_steps": self.tier_steps,
                "stopped_steady": self.stopped_steady,
                "final_steady_change": self.steady_change_last,
                "formulation_resolved": str(getattr(self.formulation, "value", self.formulation)),
                "max_strand_temperature": {
                    k: {"T": v[0], "t_relative": v[1], "x": v[2]}
                    for k, v in self.max_strand_temperature.items()
                },
                "picard_mode": "single-pass (no sub-iteration exists in the solver)",
            }
        )
        (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
