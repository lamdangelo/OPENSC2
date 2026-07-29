"""Autosave checkpoints and restart of a simulation run.

A checkpoint is a single ``np.savez`` archive of every quantity that evolves
across time steps and is not recomputed deterministically at the start of a
step: the time-integration history, the per-conductor counters and diagnostic
accumulators, the multi-level heat-source arrays and the dB/dt cache of the
solid components, the coolant primary fields, the electric solution, and the
hydraulic-network state. Everything else (material properties, heat transfer
coefficients, friction factors, Gauss-point averages, ...) is rebuilt each
step from this state, so a restarted run continues bit-identically.

Writing happens every ``AUTOSAVE_INTERVAL`` accepted time steps, after the
time-evolution buffers have been flushed to file (the caller's duty), so a
checkpoint never has to carry buffered output rows: the output files are
complete up to the checkpoint time by construction. On restart the normal
(deterministic) initialization runs first, then :func:`restore_from_checkpoint`
overwrites the evolving state and truncates the output files to the
checkpoint time, and the time loop continues as if never interrupted.

Restart is refused when the checkpoint was written by an incompatible
configuration (schema version, integration method, or - once the selectable
hydraulic formulation exists - a mismatched formulation): converting state
between formulations would be inconsistent with the stored history levels.
"""

import os
import re
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1

_CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint_step_(\d{8})\.npz$")

# Number of most recent checkpoints kept on disk; older ones are pruned
# after a new checkpoint has been written successfully.
_CHECKPOINTS_KEPT = 2

# Optional per-solid multi-level heat-source arrays (missing when the
# component never carried a transport current).
_SOLID_NODE_FIELD_NAMES = ("JHTFLX", "EEXT", "EJHT", "total_linear_power_el_cond")
_SOLID_GAUSS_FIELD_NAMES = ("linear_power_el_resistance",)
_FIELD_RATE_ATTRIBUTES = (
    "_field_rate_field_old",
    "_field_rate_time_old",
    "_field_rate_time",
    "_field_rate_value",
)


def checkpoint_file_name(step: int) -> str:
    return f"checkpoint_step_{step:08d}.npz"


def _put(state: dict, key: str, value) -> None:
    """Store ``value`` under ``key``; ``None`` values are simply skipped
    (the loader leaves the corresponding attribute at its initialization
    default, which is ``None`` for every optional quantity)."""
    if value is not None:
        state[key] = np.asarray(value)


def _collect_conductor_state(state: dict, conductor) -> None:
    prefix = f"conductor/{conductor.identifier}"
    _put(state, f"{prefix}/method", conductor.inputs.thermohydraulic_method.value)
    _put(state, f"{prefix}/solution", conductor.time_integration.solution)
    _put(state, f"{prefix}/load_vector", conductor.time_integration.load_vector)
    _put(
        state,
        f"{prefix}/adams_moulton_matrices",
        conductor.time_integration.adams_moulton_matrices,
    )
    _put(state, f"{prefix}/time_step", conductor.time_step)
    _put(state, f"{prefix}/previous_time_step", conductor.previous_time_step)
    _put(
        state,
        f"{prefix}/local_truncation_error_ratio",
        conductor.local_truncation_error_ratio,
    )
    _put(state, f"{prefix}/cond_time", np.asarray(conductor.cond_time))
    _put(state, f"{prefix}/cond_num_step", conductor.cond_num_step)
    _put(state, f"{prefix}/equation_eigenvalues", conductor.equation_eigenvalues)
    _put(state, f"{prefix}/enthalpy_balance", conductor.enthalpy_balance)
    _put(state, f"{prefix}/enthalpy_out", conductor.enthalpy_out)
    _put(state, f"{prefix}/enthalpy_inl", conductor.enthalpy_inl)
    _put(state, f"{prefix}/i_save", conductor.i_save)
    _put(state, f"{prefix}/num_step_save", conductor.num_step_save)
    _put(state, f"{prefix}/electric_time", conductor.electric_time)
    _put(state, f"{prefix}/cond_el_num_step", conductor.cond_el_num_step)
    _put(
        state,
        f"{prefix}/electric_solution",
        getattr(conductor, "electric_solution", None),
    )
    formulation = getattr(conductor, "hydraulic_formulation", None)
    if formulation is not None:
        _put(state, f"{prefix}/hydraulic_formulation", formulation.value)

    for f_comp in conductor.inventory.fluids.collection:
        fluid_prefix = f"{prefix}/fluid/{f_comp.identifier}"
        fields = f_comp.coolant.node_fields
        _put(state, f"{fluid_prefix}/velocity", fields.velocity)
        _put(state, f"{fluid_prefix}/pressure", fields.pressure)
        _put(state, f"{fluid_prefix}/temperature", fields.temperature)

    for s_comp in conductor.inventory.solids.collection:
        solid_prefix = f"{prefix}/solid/{s_comp.identifier}"
        _put(state, f"{solid_prefix}/temperature", s_comp.node_fields.temperature)
        for name in _SOLID_NODE_FIELD_NAMES:
            _put(
                state,
                f"{solid_prefix}/node/{name}",
                getattr(s_comp.node_fields, name, None),
            )
        for name in _SOLID_GAUSS_FIELD_NAMES:
            _put(
                state,
                f"{solid_prefix}/gauss/{name}",
                getattr(s_comp.gauss_fields, name, None),
            )
        for name in _FIELD_RATE_ATTRIBUTES:
            _put(state, f"{solid_prefix}{name}", getattr(s_comp, name, None))


def _collect_network_state(state: dict, network) -> None:
    _put(state, "network/node_pressure", network.node_pressure)
    _put(state, "network/branch_mass_flow", network.branch_mass_flow)
    _put(state, "network/node_temperature", network.node_temperature)
    _put(state, "network/node_temperature_rate", network.node_temperature_rate)
    _put(state, "network/solution_history", network._solution_history)
    _put(state, "network/previous_source", network._previous_source)
    _put(state, "network/last_source", network._last_source)
    _put(state, "network/num_step", network.num_step)
    _put(state, "network/previous_time_step", network.previous_time_step)
    if network.relief_valve_open:
        _put(
            state,
            "network/relief_valve_branches",
            np.asarray(sorted(network.relief_valve_open), dtype=int),
        )
        _put(
            state,
            "network/relief_valve_open",
            np.asarray(
                [network.relief_valve_open[b] for b in sorted(network.relief_valve_open)],
                dtype=bool,
            ),
        )


def write_checkpoint(simulation) -> Path:
    """Write a checkpoint of the current simulation state and prune old ones.

    The archive is written to a temporary name and moved into place
    atomically, so a run killed mid-write never corrupts the newest
    checkpoint on disk.
    """
    directory = Path(simulation.dict_path["Checkpoints_dir"])
    directory.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "manifest/schema_version": np.asarray(CHECKPOINT_SCHEMA_VERSION),
        "manifest/step": np.asarray(simulation.num_step),
        "manifest/time": np.asarray(simulation.simulation_time[-1]),
        "manifest/tend": np.asarray(simulation.transient_input["TEND"]),
    }
    for conductor in simulation.list_of_Conductors:
        _collect_conductor_state(state, conductor)
    if simulation.hydraulic_network is not None:
        _collect_network_state(state, simulation.hydraulic_network)

    target = directory / checkpoint_file_name(simulation.num_step)
    temporary = directory / (target.name + ".tmp.npz")
    with open(temporary, "wb") as stream:
        np.savez(stream, **state)
    os.replace(temporary, target)

    for stale in list_checkpoints(directory)[:-_CHECKPOINTS_KEPT]:
        stale.unlink()
    return target


def list_checkpoints(directory) -> list:
    """Checkpoint files in ``directory``, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    matches = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if (match := _CHECKPOINT_NAME_PATTERN.match(path.name))
    ]
    return [path for _, path in sorted(matches)]


def _get(archive, key: str, default=None):
    if key in archive:
        value = archive[key]
        return value.item() if value.ndim == 0 else value
    return default


def _restore_conductor_state(archive, conductor) -> None:
    prefix = f"conductor/{conductor.identifier}"
    if f"{prefix}/solution" not in archive:
        raise ValueError(
            f"Checkpoint holds no state for conductor {conductor.identifier!r}:"
            " the input deck does not match the checkpointed run."
        )
    stored_method = _get(archive, f"{prefix}/method")
    if stored_method != conductor.inputs.thermohydraulic_method.value:
        raise ValueError(
            f"Checkpoint of conductor {conductor.identifier!r} was written "
            f"with integration method {stored_method!r}, but the input deck "
            f"declares {conductor.inputs.thermohydraulic_method.value!r}; "
            "restart requires the identical method."
        )
    stored_formulation = _get(archive, f"{prefix}/hydraulic_formulation")
    active_formulation = getattr(conductor, "hydraulic_formulation", None)
    if stored_formulation is not None and active_formulation is not None:
        if stored_formulation != active_formulation.value:
            raise ValueError(
                f"Checkpoint of conductor {conductor.identifier!r} was "
                f"written with hydraulic formulation {stored_formulation!r}, "
                f"but this run resolves to {active_formulation.value!r}; "
                "state is not converted between formulations - restart with "
                "the original formulation instead."
            )

    conductor.time_integration.solution[...] = archive[f"{prefix}/solution"]
    conductor.time_integration.load_vector[...] = archive[f"{prefix}/load_vector"]
    if f"{prefix}/adams_moulton_matrices" in archive:
        conductor.time_integration.adams_moulton_matrices[...] = archive[
            f"{prefix}/adams_moulton_matrices"
        ]
    conductor.time_step = _get(archive, f"{prefix}/time_step")
    conductor.previous_time_step = _get(archive, f"{prefix}/previous_time_step")
    conductor.local_truncation_error_ratio = _get(
        archive, f"{prefix}/local_truncation_error_ratio"
    )
    conductor.cond_time = list(archive[f"{prefix}/cond_time"])
    conductor.cond_num_step = _get(archive, f"{prefix}/cond_num_step")
    conductor.equation_eigenvalues[...] = archive[f"{prefix}/equation_eigenvalues"]
    conductor.enthalpy_balance = _get(archive, f"{prefix}/enthalpy_balance")
    conductor.enthalpy_out = _get(archive, f"{prefix}/enthalpy_out")
    conductor.enthalpy_inl = _get(archive, f"{prefix}/enthalpy_inl")
    conductor.i_save = _get(archive, f"{prefix}/i_save")
    conductor.num_step_save[...] = archive[f"{prefix}/num_step_save"]
    conductor.electric_time = _get(archive, f"{prefix}/electric_time")
    conductor.cond_el_num_step = _get(archive, f"{prefix}/cond_el_num_step")
    if f"{prefix}/electric_solution" in archive:
        conductor.electric_solution = archive[f"{prefix}/electric_solution"]

    for f_comp in conductor.inventory.fluids.collection:
        fluid_prefix = f"{prefix}/fluid/{f_comp.identifier}"
        fields = f_comp.coolant.node_fields
        fields.velocity = archive[f"{fluid_prefix}/velocity"].copy()
        fields.pressure = archive[f"{fluid_prefix}/pressure"].copy()
        fields.temperature = archive[f"{fluid_prefix}/temperature"].copy()

    for s_comp in conductor.inventory.solids.collection:
        solid_prefix = f"{prefix}/solid/{s_comp.identifier}"
        s_comp.node_fields.temperature = archive[
            f"{solid_prefix}/temperature"
        ].copy()
        for name in _SOLID_NODE_FIELD_NAMES:
            if f"{solid_prefix}/node/{name}" in archive:
                getattr(s_comp.node_fields, name)[...] = archive[
                    f"{solid_prefix}/node/{name}"
                ]
        for name in _SOLID_GAUSS_FIELD_NAMES:
            if f"{solid_prefix}/gauss/{name}" in archive:
                getattr(s_comp.gauss_fields, name)[...] = archive[
                    f"{solid_prefix}/gauss/{name}"
                ]
        for name in _FIELD_RATE_ATTRIBUTES:
            if f"{solid_prefix}{name}" in archive:
                setattr(s_comp, name, _get(archive, f"{solid_prefix}{name}"))


def _restore_network_state(archive, network) -> None:
    if "network/node_pressure" not in archive:
        raise ValueError(
            "The input deck declares a hydraulic network but the checkpoint "
            "holds no network state: it does not match the checkpointed run."
        )
    network.node_pressure[...] = archive["network/node_pressure"]
    network.branch_mass_flow[...] = archive["network/branch_mass_flow"]
    network.node_temperature[...] = archive["network/node_temperature"]
    network.node_temperature_rate[...] = archive["network/node_temperature_rate"]
    network._solution_history[...] = archive["network/solution_history"]
    network._previous_source = _get(archive, "network/previous_source")
    network._last_source = _get(archive, "network/last_source")
    network.num_step = _get(archive, "network/num_step")
    network.previous_time_step = _get(archive, "network/previous_time_step")
    if "network/relief_valve_branches" in archive:
        branches = archive["network/relief_valve_branches"]
        opened = archive["network/relief_valve_open"]
        for branch, is_open in zip(branches, opened):
            network.relief_valve_open[int(branch)] = bool(is_open)


def restore_from_checkpoint(simulation) -> float:
    """Overwrite the freshly initialized simulation state from the newest
    checkpoint and truncate the output files to the checkpoint time.

    Returns the checkpoint time. Must run after ``conductor_initialization``
    (which builds every object and resets the counters this function
    overwrites) and before ``conductor_solution``.
    """
    directory = Path(simulation.dict_path["Checkpoints_dir"])
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        raise FileNotFoundError(
            f"RESTART is set but no checkpoint exists under {directory}. "
            "Run with autosave_interval first, or unset restart."
        )
    newest = checkpoints[-1]
    with np.load(newest, allow_pickle=False) as archive:
        schema_version = _get(archive, "manifest/schema_version")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"{newest.name}: checkpoint schema version {schema_version} "
                f"does not match this code ({CHECKPOINT_SCHEMA_VERSION})."
            )
        checkpoint_time = _get(archive, "manifest/time")
        simulation.num_step = _get(archive, "manifest/step")
        simulation.simulation_time = [checkpoint_time]
        for conductor in simulation.list_of_Conductors:
            _restore_conductor_state(archive, conductor)
        if simulation.hydraulic_network is not None:
            _restore_network_state(archive, simulation.hydraulic_network)

    for conductor in simulation.list_of_Conductors:
        _reject_finalized_run(simulation, conductor)
        _truncate_conductor_outputs(simulation, conductor, checkpoint_time)

    print(
        f"Restart: resuming from {newest.name} "
        f"(t = {checkpoint_time} s, step {simulation.num_step}).\n"
    )
    return checkpoint_time


# --------------------------------------------------------------------------
# Output-file truncation
# --------------------------------------------------------------------------

_SPATIAL_SAVE_STEP_PATTERN = re.compile(r"_\((\d+)\)_(?:gauss_)?sd\.tsv$")


def _reject_finalized_run(simulation, conductor) -> None:
    """Refuse to restart a run whose post-processing already completed.

    ``reorganize_spatial_distribution`` consumes (deletes) the raw per-step
    spatial-distribution files, so a finalized run cannot reproduce them and
    a restarted continuation would produce incomplete reorganized outputs.
    Restart targets interrupted runs (crash, kill, out-of-walltime), where
    post-processing never ran and the raw files are still on disk.
    """
    if conductor.inventory.fluids.collection:
        reference_component = conductor.inventory.fluids.collection[0]
    else:
        reference_component = conductor.inventory.solids.collection[0]
    spatial_dir = Path(
        simulation.dict_path[
            f"Output_Spatial_distribution_{conductor.identifier}_dir"
        ]
    )
    for save_index in range(conductor.i_save):
        step_number = conductor.num_step_save[save_index]
        raw_file = spatial_dir / (
            f"{reference_component.identifier}_({step_number})_sd.tsv"
        )
        if not raw_file.is_file():
            raise RuntimeError(
                f"Cannot restart conductor {conductor.identifier!r}: the raw "
                f"spatial save {raw_file.name} is missing - this run was "
                "already finalized (its spatial saves were reorganized by "
                "the post-processing). Restart is meant for interrupted "
                "runs; rerun from scratch instead."
            )


def _truncate_conductor_outputs(simulation, conductor, checkpoint_time) -> None:
    """Drop output content produced after the checkpoint time.

    Time-evolution files get every data row with time > checkpoint time
    removed (line-based, so the kept prefix stays byte-identical); spatial
    distribution files stamped with a step counter beyond the checkpoint are
    deleted (the resumed run regenerates them at the same names).
    """
    time_evolution_dir = simulation.dict_path[
        f"Output_Time_evolution_{conductor.identifier}_dir"
    ]
    for path in sorted(Path(time_evolution_dir).glob("*.tsv")):
        _truncate_time_column_file(path, checkpoint_time)

    spatial_dir = Path(
        simulation.dict_path[
            f"Output_Spatial_distribution_{conductor.identifier}_dir"
        ]
    )
    for path in sorted(spatial_dir.glob("*.tsv")):
        match = _SPATIAL_SAVE_STEP_PATTERN.search(path.name)
        if match and int(match.group(1)) > conductor.cond_num_step:
            path.unlink()
    _truncate_time_column_file(
        spatial_dir / "Time_sd_actual.tsv", checkpoint_time
    )


def _truncate_time_column_file(path, checkpoint_time) -> None:
    """Remove trailing rows with time > checkpoint_time from a tsv whose
    first column is a time in seconds; files without such a column (or
    absent files) are left untouched. Kept lines are preserved verbatim."""
    path = Path(path)
    if not path.is_file():
        return
    lines = path.read_text().splitlines(keepends=True)
    kept = []
    for line_number, line in enumerate(lines):
        first_field = line.split("\t", 1)[0].strip()
        try:
            time_value = float(first_field)
        except ValueError:
            if line_number == 0:
                kept.append(line)  # header
                continue
            return  # not a time-column file: leave untouched
        if time_value > checkpoint_time * (1.0 + 1e-12):
            break
        kept.append(line)
    if len(kept) < len(lines):
        path.write_text("".join(kept))
