"""Step callback of the winding-pack (WP) quench-test runs.

The controller replays the CEA Saclay quench test of coil AAB51 on the
six-double-layer model:

* the supply reservoir of the hydraulic network follows the measured
  PT2111 / TT2113C traces (pressure and temperature at the coil supply)
  with event re-anchoring: before the model detects a quench the
  schedule is clamped at measured time -2 s, afterwards the measured
  event window is replayed anchored to the MODEL detection (the real
  detection sits at +0.461 s);
* the quench is detected when the resistive voltage of any double layer
  reaches 0.5 V; 0.184 s later the coil current is switched to the
  measured IAL-C1 dump waveform;
* the run stops a fixed time after the dump starts;
* every step the double-layer voltages, the outlet group flows and the
  spatial hot spots (strand, jacket, epoxy of every double layer) are
  logged to ``final_run_log.tsv`` and ``hotspot_log.tsv`` in the model
  directory (the figure scripts read both).

The induced-loss source terms are configured at start-up: with
``induced_losses=True`` every strand's coupling-loss coefficient n*tau is
set to ``coupling_tau``; with ``induced_losses=False`` coupling, eddy and
hysteresis losses are zeroed on every strand and jacket.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import measurement_data as md
from .constants import (
    BAR,
    DETECTION_THRESHOLD,
    DETECTION_TO_DISCHARGE_DELAY,
    EVENT_WINDOW_START,
    MEASURED_DETECTION_TIME,
    POST_DISCHARGE_DURATION,
    SIMULATION_START_MEASURED_TIME,
)

OUTLET_GROUPS = (1, 2, 3, 4)
SOLID_KINDS = ("strand", "jacket", "epoxy")


class QuenchTestController:
    def __init__(self, model_directory: Path, apply_schedules: bool = True,
                 post_discharge_duration: float = POST_DISCHARGE_DURATION,
                 induced_losses: bool = True,
                 coupling_tau: float = 58e-3):
        self.model_directory = Path(model_directory)
        self.apply_schedules = apply_schedules
        self.post_discharge_duration = post_discharge_duration
        self.induced_losses = induced_losses
        self.coupling_tau = coupling_tau
        # True spatial hot-spot evolution (max over ALL mesh nodes, per
        # DL and solid kind), logged every step: the Output *_te.tsv
        # probes only sample a few fixed z-stations and under-read the
        # peak.
        self.hotspot_file = self.model_directory / "hotspot_log.tsv"
        self.hotspot_rows = []
        self.log_file = self.model_directory / "final_run_log.tsv"
        self.log_rows = []
        self.initialized = False
        self.detection_time = None
        self.discharge_time = None
        self.discharge_started = False

        if apply_schedules:
            self.supply_pressure = self._interpolant("PT2111_ai", "PT2111", BAR)
            self.supply_temperature = self._interpolant(
                "TT2113C_ai", "TT2113C", 1.0
            )
        self.template_time, self.template_current = (
            md.discharge_current_template()
        )

    def _interpolant(self, slow_sensor: str, event_sensor: str,
                     scale: float):
        measured_time, value = md.merged_schedule(
            slow_sensor, event_sensor, scale=scale, window=(-1.4e3, 9.0)
        )

        def evaluate(simulation_time: float) -> float:
            if self.detection_time is None:
                query = min(
                    simulation_time + SIMULATION_START_MEASURED_TIME,
                    EVENT_WINDOW_START,
                )
            else:
                query = MEASURED_DETECTION_TIME + (
                    simulation_time - self.detection_time
                )
            return float(np.interp(query, measured_time, value))

        return evaluate

    def _initialize(self, simulation):
        self.double_layer_conductors = {}
        self.strands = []
        for conductor in simulation.list_of_Conductors:
            double_layer = int(conductor.identifier.split("_")[1]) // 100
            self.double_layer_conductors.setdefault(double_layer, []).append(
                conductor
            )
            self.strands.extend(conductor.inventory.strands.collection)
        self.initial_current = float(
            simulation.list_of_Conductors[0].inputs.initial_current
        )
        # Per-DL solid components for the hot-spot log, keyed by the
        # report kind (strand / jacket / epoxy).
        self.solid_components = {}
        for double_layer, conductors in self.double_layer_conductors.items():
            groups = {kind: [] for kind in SOLID_KINDS}
            for conductor in conductors:
                groups["strand"].extend(
                    conductor.inventory.strands.collection
                )
                for jacket in conductor.inventory.jackets.collection:
                    kind = (
                        "jacket"
                        if jacket.identifier.startswith("Z_JACKET1")
                        else "epoxy"
                    )
                    groups[kind].append(jacket)
            self.solid_components[double_layer] = groups
        # Induced-loss configuration (coupling n*tau / eddy / hysteresis).
        touched_strands = 0
        for conductor in simulation.list_of_Conductors:
            for strand in conductor.inventory.strands.collection:
                if not self.induced_losses:
                    strand.operations.coupling_loss_time_constant = 0.0
                    strand.operations.eddy_loss_geometry_constant = 0.0
                    strand.operations.filament_diameter = 0.0
                else:
                    strand.operations.coupling_loss_time_constant = (
                        self.coupling_tau
                    )
                touched_strands += 1
            if not self.induced_losses:
                for jacket in conductor.inventory.jackets.collection:
                    jacket.operations.eddy_loss_geometry_constant = 0.0
        if not self.induced_losses:
            print(
                "[controller] induced losses disabled (coupling, eddy, "
                "hysteresis zeroed on all strands and jackets)"
            )
        else:
            print(
                f"[controller] coupling loss n·tau = "
                f"{self.coupling_tau:.4e} s on {touched_strands} strands"
            )
        self.initialized = True

    def double_layer_voltage(self, double_layer: int) -> float:
        voltage = 0.0
        for conductor in self.double_layer_conductors[double_layer]:
            for strand in conductor.inventory.strands.collection:
                resistance = strand.gauss_fields.electric_resistance
                current = getattr(strand.gauss_fields, "current_along", None)
                if resistance is None:
                    continue
                if current is None:
                    voltage += float(np.sum(resistance)) * self.initial_current
                else:
                    voltage += float(np.sum(resistance * current))
        return voltage

    def switch_to_discharge(self, time_now: float):
        from utility_functions.auxiliary_functions import build_interpolator

        times = np.concatenate(
            (
                [0.0, self.discharge_time],
                self.discharge_time + self.template_time[1:],
            )
        )
        currents = np.concatenate(
            (
                [self.initial_current, self.initial_current],
                self.template_current[1:],
            )
        )
        frame = pd.DataFrame(
            np.vstack(
                (
                    np.concatenate(([0.0], times)),
                    np.concatenate(([0.0], currents)),
                )
            )
        )
        for strand in self.strands:
            (
                strand.current_interpolator,
                strand.current_interp_flag,
            ) = build_interpolator(
                frame, strand.operations.operating_current_interpolation
            )
        self.discharge_started = True
        print(
            f"[controller] discharge started at t = {time_now:.4f} s "
            f"(scheduled {self.discharge_time:.4f} s)"
        )

    def __call__(self, simulation) -> bool:
        if not self.initialized:
            self._initialize(simulation)
        time_now = simulation.simulation_time[-1]

        network = simulation.hydraulic_network
        if self.apply_schedules and network is not None:
            network.set_reservoir_state(
                "supply",
                pressure=self.supply_pressure(time_now),
                temperature=self.supply_temperature(time_now),
            )

        voltages = {
            double_layer: self.double_layer_voltage(double_layer)
            for double_layer in sorted(self.double_layer_conductors)
        }
        if self.detection_time is None:
            if any(
                abs(voltage) >= DETECTION_THRESHOLD
                for voltage in voltages.values()
            ):
                self.detection_time = time_now
                self.discharge_time = time_now + DETECTION_TO_DISCHARGE_DELAY
                print(
                    f"[controller] detection at t = {time_now:.4f} s "
                    f"(t_meas {time_now + SIMULATION_START_MEASURED_TIME:+.1f}"
                    " s; voltages: "
                    + ", ".join(
                        f"DL{dl} {v:.3f} V" for dl, v in voltages.items()
                    )
                    + ")"
                )
        elif not self.discharge_started and time_now >= self.discharge_time:
            self.switch_to_discharge(time_now)

        flows = [
            network.branch_mass_flow_of(f"outlet{group}_to_return")
            if network is not None
            else 0.0
            for group in OUTLET_GROUPS
        ]
        self.log_rows.append(
            (
                time_now,
                *(voltages[dl] for dl in sorted(voltages)),
                *flows,
            )
        )
        self.hotspot_rows.append(
            (
                time_now,
                *(
                    max(
                        float(np.max(component.node_fields.temperature))
                        for component in (
                            self.solid_components[double_layer][kind]
                        )
                    )
                    for double_layer in sorted(self.solid_components)
                    for kind in SOLID_KINDS
                ),
            )
        )
        if len(self.log_rows) % 200 == 0:
            self.flush_log()

        if (
            self.discharge_started
            and time_now >= self.discharge_time + self.post_discharge_duration
        ):
            print(f"[controller] stopping at t = {time_now:.4f} s")
            self.flush_log()
            return True
        return False

    def flush_log(self):
        if not self.log_rows:
            return
        labels = (
            ["time"]
            + [f"V_DL{dl}" for dl in sorted(self.double_layer_conductors)]
            + [f"mdot_group_{group}" for group in OUTLET_GROUPS]
        )
        pd.DataFrame(self.log_rows, columns=labels).to_csv(
            self.log_file, sep="\t", index=False
        )
        if self.hotspot_rows:
            hotspot_labels = ["time"] + [
                f"{kind}_max_DL{double_layer}"
                for double_layer in sorted(self.solid_components)
                for kind in SOLID_KINDS
            ]
            pd.DataFrame(self.hotspot_rows, columns=hotspot_labels).to_csv(
                self.hotspot_file, sep="\t", index=False
            )
