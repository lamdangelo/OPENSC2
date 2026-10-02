"""End-of-run simulation summary and hot-spot tracking.

Two output files are produced from a single per-step tracker
(:class:`SimulationSummary`), both purely observational (read-only on the
solver state):

* ``simulation.log`` — human-readable end-of-run summary written next to the
  ``Output`` folder of the run: metadata (timestamp, model, runtime, steps,
  transient window), per-conductor extrema of temperature, pressure,
  velocity, mass flow rate and voltage with the time and location at which
  they occurred, the resolved hydraulic formulation and a checklist of the
  optional physical features active in the run.
* ``hotspot_log.tsv`` — per-conductor time series of the spatial maximum
  temperature of every component (one row per time step), written to the
  conductor's ``Output/Time_evolution`` folder. The fixed-coordinate probes
  of the ``*_te.tsv`` files under-read a travelling peak; this file records
  the true spatial maximum at every step. Rows are appended in chunks, so
  the restart machinery's time-column truncation applies to it like to any
  other time-evolution file.

The extrema are tracked per component; the log reports one line per
conductor per quantity, annotated with the component that reached it.
"""

from dataclasses import dataclass
from datetime import datetime
import os
import time
from typing import Optional

import numpy as np

from electromagnetics.electromagnetic_flags import CurrentMode

# Rows buffered per conductor before the hot-spot file is appended to.
HOTSPOT_FLUSH_INTERVAL = 200

# Display rows of the per-conductor extrema section: label, record key,
# unit, and whether the record carries a location.
EXTREMA_LAYOUT = (
    ("max temperature", "max_temperature", "K", True),
    ("max pressure", "max_pressure", "Pa", True),
    ("max velocity", "max_velocity", "m/s", True),
    ("min velocity", "min_velocity", "m/s", True),
    ("max mass flow rate", "max_mass_flow_rate", "kg/s", True),
    ("min mass flow rate", "min_mass_flow_rate", "kg/s", True),
    ("max |tap voltage|", "max_abs_tap_voltage", "V", False),
    ("max |end-to-end voltage|", "max_abs_end_to_end_voltage", "V", False),
)

VOLTAGE_RECORD_KEYS = ("max_abs_tap_voltage", "max_abs_end_to_end_voltage")


@dataclass
class ExtremumRecord:
    value: float
    time: float  # s
    location: Optional[float]  # m; None when not meaningful (voltage)


class ComponentExtrema:
    """Running extrema records of a single component."""

    def __init__(self, identifier: str, kind: str):
        self.identifier = identifier
        self.kind = kind  # "fluid" | "solid"
        self.records = {}  # record key -> ExtremumRecord

    def observe_max(self, name, values, time_s, coordinates=None) -> float:
        """Update the running maximum record ``name``; return the current
        spatial maximum (reused by the hot-spot time series)."""
        return self._observe(name, values, time_s, coordinates, minimum=False)

    def observe_min(self, name, values, time_s, coordinates=None) -> float:
        return self._observe(name, values, time_s, coordinates, minimum=True)

    def _observe(self, name, values, time_s, coordinates, minimum) -> float:
        array = np.ravel(values)
        index = int(np.argmin(array) if minimum else np.argmax(array))
        value = float(array[index])
        record = self.records.get(name)
        if record is None or (value < record.value if minimum else value > record.value):
            location = None if coordinates is None else float(coordinates[index])
            self.records[name] = ExtremumRecord(value, time_s, location)
        return value


def conductor_level_extremum(components, record_name, minimum=False):
    """Extremum of ``record_name`` over a conductor's components.

    Args:
        components: iterable of ComponentExtrema.
        record_name: record key, e.g. ``"max_temperature"``.
        minimum: pick the smallest record value instead of the largest.

    Returns:
        (component identifier, ExtremumRecord) of the winning component, or
        None when no component carries the record.
    """
    best = None
    for component in components:
        record = component.records.get(record_name)
        if record is None:
            continue
        if (
            best is None
            or (record.value < best[1].value if minimum else record.value > best[1].value)
        ):
            best = (component.identifier, record)
    return best


def _solid_components(conductor):
    yield from conductor.inventory.strands.collection
    yield from conductor.inventory.jackets.collection


def normal_zone_length(strand, element_lengths) -> Optional[float]:
    """Length of the strand's normal/current-sharing zone in meters.

    Uses the regime partition stored by ``get_electric_resistance``
    (superconducting strands only): an element counts as normal when its
    resistance-evaluation current is at or above 0.95 of the local,
    temperature-dependent critical current — the same criterion the
    electric module uses for the regime split. Returns None for strands
    without regime data (pure stabilizers, or electric module inactive).
    """
    regimes = getattr(strand, "_electric_regime_gauss", None)
    if regimes is None:
        return None
    indices = np.concatenate((regimes["sharing"], regimes["normal"]))
    if indices.size == 0:
        return 0.0
    return float(np.sum(np.asarray(element_lengths)[indices.astype(int)]))


def estimate_normal_zone_propagation(times, lengths,
                                     disturbance_end=0.0) -> Optional[dict]:
    """NZPV estimate from a normal-zone-length time series.

    The front velocity is half the growth rate of the zone length (two
    fronts propagate symmetrically), measured between the times at which
    the post-disturbance growth first reaches 25 % and 75 % of its
    maximum — a rise-time style window that skips the saturated/dump
    phase. Everything up to ``disturbance_end`` (the external-heater
    switch-off time) is excluded from the velocity: a heater IGNITES its
    zone near-instantaneously, and measuring that as "propagation"
    produces absurd velocities. Only growth beyond the zone length left
    at heater-off counts as front propagation.

    Returns None when the zone never formed; the "velocity" key is None
    when the zone never grew beyond the disturbance-made length.
    """
    times = np.asarray(times, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    formed = np.nonzero(lengths > 0.0)[0]
    if formed.size == 0:
        return None
    peak_index = int(np.argmax(lengths))
    result = {
        "onset_time": float(times[formed[0]]),
        "max_length": float(lengths[peak_index]),
        "max_length_time": float(times[peak_index]),
        "velocity": None,
        "window": None,
    }

    after = np.nonzero(times >= disturbance_end)[0]
    if after.size == 0:
        return result
    times_after = times[after]
    lengths_after = lengths[after]
    baseline = float(lengths_after[0])
    peak_after = int(np.argmax(lengths_after))
    # Propagation requires growth beyond the disturbance-made zone (5 %
    # margin against regime-boundary flicker).
    if lengths_after[peak_after] <= max(baseline * 1.05, baseline + 1e-12):
        return result
    growth = lengths_after[: peak_after + 1] - baseline
    target = growth[peak_after]
    lower = np.nonzero(growth >= 0.25 * target)[0]
    upper = np.nonzero(growth >= 0.75 * target)[0]
    if lower.size and upper.size:
        t1 = float(times_after[lower[0]])
        t2 = float(times_after[upper[0]])
        if t2 > t1:
            result["velocity"] = (
                0.5 * (growth[upper[0]] - growth[lower[0]]) / (t2 - t1)
            )
            result["window"] = (t1, t2)
    return result


def disturbance_end_time(conductor) -> float:
    """Latest external-heater switch-off time on the conductor (0.0 when
    no component carries a square-wave heat impulse)."""
    end = 0.0
    for solid in _solid_components(conductor):
        operations = getattr(solid, "operations", None)
        if operations is None:
            continue
        if float(getattr(operations, "heat_flux_amplitude", 0.0) or 0.0) > 0.0:
            end = max(
                end, float(getattr(operations, "heat_flux_time_end", 0.0))
            )
    return end


def _positive(value) -> bool:
    """True when a scalar, or any entry of a list of loop families, exceeds 0."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_positive(v) for v in value)
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def detect_features(conductor):
    """Checklist of the optional physical features active on a conductor."""
    solids = list(_solid_components(conductor))
    strands = list(conductor.inventory.strands.collection)
    return {
        "hydraulic network": bool(getattr(conductor, "network_ports", [])),
        "coupling loss": any(
            _positive(getattr(s.operations, "coupling_loss_time_constant", 0.0))
            for s in strands
        ),
        "eddy current loss": any(
            getattr(c.operations, "eddy_loss_geometry_constant", 0.0) > 0.0
            for c in solids
        ),
        "hysteresis loss": any(
            _positive(getattr(s.operations, "filament_diameter", 0.0))
            or bool(getattr(s.operations, "tape_hysteresis_loss", False))
            for s in strands
        ),
        "casing bath (BOUNDARY sink)": any(
            patch["partner"] is None
            for c in solids
            for patch in (getattr(c, "_transverse_patches", None) or [])
        ),
        "transversal thermal coupling": any(
            getattr(c.operations, "transverse_coupling_file", "") != ""
            for c in solids
        ),
        "B ~ I scaling": any(
            getattr(c.operations, "magnetic_field_scales_with_current", False)
            for c in solids
        ),
    }


class SimulationSummary:
    """Per-step tracker feeding ``simulation.log`` and ``hotspot_log.tsv``."""

    def __init__(self):
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        # conductor identifier -> component identifier -> ComponentExtrema
        self._per_conductor = {}
        # conductor identifier -> strand identifier -> {"time": [], "length": []}
        self._normal_zone = {}
        # conductor identifier -> buffered hot-spot rows / column labels
        self._hotspot_rows = {}
        self._hotspot_columns = {}
        # Conductors whose hot-spot file was already flushed by this run.
        self._flushed_hotspot_conductors = set()

    # -- per-step tracking ------------------------------------------------

    def update(self, simulation, conductor, record_hotspot=True) -> None:
        """Refresh the running extrema and the hot-spot buffer of one
        conductor from its current fields. Pure reads.

        Args:
            record_hotspot: False when re-seeding from a restored
                checkpoint, whose hot-spot row is already in the truncated
                file.
        """
        time_now = conductor.cond_time[-1]
        coordinates = conductor.mesh.node_coordinates
        per_component = self._per_conductor.setdefault(conductor.identifier, {})
        hotspot_row = [time_now]
        hotspot_columns = ["time (s)"]

        for fluid_comp in conductor.inventory.fluids.collection:
            fields = fluid_comp.coolant.node_fields
            extrema = per_component.setdefault(
                fluid_comp.identifier,
                ComponentExtrema(fluid_comp.identifier, "fluid"),
            )
            hot = extrema.observe_max(
                "max_temperature", fields.temperature, time_now, coordinates
            )
            extrema.observe_max("max_pressure", fields.pressure, time_now, coordinates)
            extrema.observe_max("max_velocity", fields.velocity, time_now, coordinates)
            extrema.observe_min("min_velocity", fields.velocity, time_now, coordinates)
            extrema.observe_max(
                "max_mass_flow_rate", fields.mass_flow_rate, time_now, coordinates
            )
            extrema.observe_min(
                "min_mass_flow_rate", fields.mass_flow_rate, time_now, coordinates
            )
            hotspot_row.append(hot)
            hotspot_columns.append(f"{fluid_comp.identifier}_max_temperature")

        electric_on = (
            conductor.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED
        )
        n_strands = conductor.inventory.strands.number
        nodal_potential = getattr(conductor, "nodal_potential", None)

        element_lengths = getattr(conductor.mesh, "element_lengths", None)
        for solid in _solid_components(conductor):
            extrema = per_component.setdefault(
                solid.identifier, ComponentExtrema(solid.identifier, "solid")
            )
            hot = extrema.observe_max(
                "max_temperature", solid.node_fields.temperature, time_now, coordinates
            )
            hotspot_row.append(hot)
            hotspot_columns.append(f"{solid.identifier}_max_temperature")
            if electric_on and element_lengths is not None:
                zone_length = normal_zone_length(solid, element_lengths)
                if zone_length is not None:
                    hotspot_row.append(zone_length)
                    hotspot_columns.append(
                        f"{solid.identifier}_normal_zone_length"
                    )
                    if record_hotspot:
                        series = self._normal_zone.setdefault(
                            conductor.identifier, {}
                        ).setdefault(
                            solid.identifier, {"time": [], "length": []}
                        )
                        series["time"].append(time_now)
                        series["length"].append(zone_length)

        if electric_on:
            for ii, strand in enumerate(conductor.inventory.strands.collection):
                extrema = per_component[strand.identifier]
                if strand.gauss_fields.has("delta_voltage_along_sum"):
                    extrema.observe_max(
                        "max_abs_tap_voltage",
                        np.abs(strand.gauss_fields.delta_voltage_along_sum),
                        time_now,
                    )
                if nodal_potential is not None:
                    # Nodal potentials are interleaved by strand (see
                    # electromagnetics/electric_solver.py).
                    potential = nodal_potential[ii::n_strands]
                    extrema.observe_max(
                        "max_abs_end_to_end_voltage",
                        abs(potential[-1] - potential[0]),
                        time_now,
                    )

        if record_hotspot:
            rows = self._hotspot_rows.setdefault(conductor.identifier, [])
            rows.append(hotspot_row)
            self._hotspot_columns[conductor.identifier] = hotspot_columns
            if len(rows) >= HOTSPOT_FLUSH_INTERVAL:
                self._flush_hotspot(simulation, conductor.identifier)

    # -- hot-spot file ----------------------------------------------------

    def _hotspot_path(self, simulation, conductor_identifier) -> str:
        return os.path.join(
            simulation.dict_path[
                f"Output_Time_evolution_{conductor_identifier}_dir"
            ],
            "hotspot_log.tsv",
        )

    def _flush_hotspot(self, simulation, conductor_identifier) -> None:
        rows = self._hotspot_rows.get(conductor_identifier)
        if not rows:
            return
        path = self._hotspot_path(simulation, conductor_identifier)
        # A fresh run truncates any file left by a previous run of the same
        # simulation name on its first flush (matching the header rewrite of
        # the other time-evolution files); a restarted run appends to the
        # checkpoint-truncated file instead.
        truncate = (
            conductor_identifier not in self._flushed_hotspot_conductors
            and not simulation.transient_input.get("RESTART")
        )
        write_header = truncate or not os.path.isfile(path)
        with open(path, "w" if truncate else "a") as stream:
            if write_header:
                stream.write(
                    "\t".join(self._hotspot_columns[conductor_identifier]) + "\n"
                )
            for row in rows:
                stream.write("\t".join(f"{value:.10e}" for value in row) + "\n")
        self._flushed_hotspot_conductors.add(conductor_identifier)
        self._hotspot_rows[conductor_identifier] = []

    def flush_hotspots(self, simulation) -> None:
        """Flush every buffered hot-spot row (autosave and end of run)."""
        for conductor_identifier in list(self._hotspot_rows):
            self._flush_hotspot(simulation, conductor_identifier)

    # -- end-of-run summary -----------------------------------------------

    def write(self, simulation) -> None:
        """Write ``simulation.log`` and final-flush the hot-spot buffers."""
        self.flush_hotspots(simulation)
        text = format_log(
            simulation,
            self._per_conductor,
            cpu_seconds=time.process_time() - self.cpu_start,
            wall_seconds=time.perf_counter() - self.wall_start,
            normal_zone=self._normal_zone,
        )
        path = os.path.join(
            simulation.dict_path["Sub_dir"],
            simulation.transient_input["SIMULATION"],
            "simulation.log",
        )
        with open(path, "w") as stream:
            stream.write(text)
        print(f"Saved simulation summary: {path}\n")


def _format_extremum_line(label, record, unit, component_identifier):
    where = f"at t = {record.time:.4f} s"
    if record.location is not None:
        where += f", x = {record.location:.4f} m"
    return (
        f"  {label:<24} : {record.value:>13.6e} {unit:<5} "
        f"{where:<32} ({component_identifier})"
    )


def format_log(simulation, per_conductor, cpu_seconds, wall_seconds,
               normal_zone=None) -> str:
    """Render the full simulation.log text. Pure function of its inputs."""
    transient_input = simulation.transient_input
    rule = "=" * 80
    lines = [
        rule,
        " OPENSC2 SIMULATION SUMMARY",
        rule,
        f"Simulation ended        : {datetime.now().strftime('%d.%m.%y : %H:%M:%S')}",
        f"Model (SIMULATION)      : {transient_input['SIMULATION']}",
        f"Input deck (MAGNET)     : {transient_input['MAGNET']}",
        f"CPU time                : {cpu_seconds:.2f} s",
        f"Wall-clock time         : {wall_seconds:.2f} s",
        f"Total time steps        : {simulation.num_step}",
        (
            f"Transient               : 0.0 s -> {simulation.simulation_time[-1]:.6g} s"
            f"   (requested TEND = {transient_input['TEND']:.6g} s)"
        ),
        (
            f"Time adaptivity         : IADAPTIME = {transient_input['IADAPTIME']}"
            f"   (STPMIN = {transient_input['STPMIN']:.6g} s,"
            f" STPMAX = {transient_input['STPMAX']:.6g} s)"
        ),
    ]

    for conductor in simulation.list_of_Conductors:
        components = per_conductor.get(conductor.identifier, {})
        title = f" {conductor.identifier}"
        lines += [
            "",
            "-" * (80 - len(title)) + title,
            f"Integration scheme      : {conductor.inputs.thermohydraulic_method.name}",
            f"Hydraulic formulation   : {conductor.hydraulic_formulation.value}",
            (
                f"Mesh                    : {conductor.mesh.number_of_elements} elements,"
                f" {conductor.mesh.number_of_nodes} nodes"
            ),
            "",
            "Extrema",
        ]
        for label, record_name, unit, _ in EXTREMA_LAYOUT:
            best = conductor_level_extremum(
                components.values(), record_name, minimum=record_name.startswith("min")
            )
            if best is not None:
                lines.append(_format_extremum_line(label, best[1], unit, best[0]))
            elif record_name in VOLTAGE_RECORD_KEYS:
                lines.append(
                    f"  {label:<24} : n/a (electric module disabled)"
                )
        lines += ["", "Features"]
        for feature, active in detect_features(conductor).items():
            lines.append(f"  {feature:<29}: {str(active).lower()}")

        conductor_zones = (normal_zone or {}).get(conductor.identifier, {})
        zone_lines = []
        disturbance_end = disturbance_end_time(conductor)
        for strand_identifier, series in conductor_zones.items():
            estimate = estimate_normal_zone_propagation(
                series["time"], series["length"],
                disturbance_end=disturbance_end,
            )
            if estimate is None:
                continue
            zone_lines.append(
                f"  {strand_identifier:<12}: onset t = "
                f"{estimate['onset_time']:.4f} s, max length "
                f"{estimate['max_length']:.3f} m at t = "
                f"{estimate['max_length_time']:.4f} s"
            )
            if estimate["velocity"] is not None:
                t1, t2 = estimate["window"]
                zone_lines.append(
                    f"  {'':<12}  NZPV ~ {estimate['velocity']:.3f} m/s per "
                    f"front (25-75 % growth, t = {t1:.4f} s -> {t2:.4f} s)"
                )
            else:
                zone_lines.append(
                    f"  {'':<12}  no front propagation beyond the "
                    f"disturbance-made zone (heater off at "
                    f"t = {disturbance_end:.4f} s)"
                )
        if conductor_zones:
            lines += ["", "Normal zone"]
            lines += zone_lines if zone_lines else ["  none formed"]

    if transient_input.get("RESTART"):
        lines += [
            "",
            "NOTE: restarted run - runtime and extrema cover the resumed"
            " segment only.",
        ]
    lines += [rule, ""]
    return "\n".join(lines)
