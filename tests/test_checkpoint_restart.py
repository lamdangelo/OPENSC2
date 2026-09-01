"""Autosave-checkpoint and restart tests.

The acceptance criterion is bit-identical resumption: a run interrupted by a
simulated crash and restarted from its newest autosaved checkpoint must
produce an Output tree whose every file is byte-identical to the one of an
uninterrupted run. Both the plain CASE_1 scenario and the network-coupled
variant are exercised (the latter covers the hydraulic-network state,
including the staggered node-temperature history and the theta/BDF2 source
history).

The crash is simulated by raising from the per-step driver callback: this
skips the final buffer flush and the whole post-processing, exactly like a
kill would, leaving the te files complete up to the last checkpoint (the
autosave hook flushes before writing) and the raw spatial saves in place.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from simulation import Simulation
from interfaces import yaml_input_registry
from interfaces.yaml_input_registry import (
    _parse_autosave_interval,
    _parse_restart_flag,
)

from test_network_coupling import (
    NETWORK_SECTION,
    prepare_run_directory,
    run_simulation,
)


# --------------------------------------------------------------------------
# Settings parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("none", None), ("NONE", None), (1, 1), (50, 50)],
)
def test_autosave_interval_accepts(value, expected):
    assert _parse_autosave_interval(value) == expected


@pytest.mark.parametrize("value", [0, -3, 2.5, "abc", True])
def test_autosave_interval_rejects(value):
    with pytest.raises(ValueError, match="autosave_interval"):
        _parse_autosave_interval(value)


@pytest.mark.parametrize("value", [True, False])
def test_restart_flag_accepts(value):
    assert _parse_restart_flag(value) is value


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_restart_flag_rejects(value):
    with pytest.raises(ValueError, match="restart"):
        _parse_restart_flag(value)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def set_simulation_keys(run_directory: Path, **keys) -> None:
    """Update simulation-level keys of the copied deck and drop the cached
    registry (a rerun of the same directory must reparse the file)."""
    simulation_file = run_directory / "simulation.yaml"
    document = yaml.safe_load(simulation_file.read_text())
    document["simulation"].update(keys)
    simulation_file.write_text(yaml.safe_dump(document, sort_keys=False))
    yaml_input_registry._registry_cache.clear()


def output_tree_hashes(run_directory: Path) -> dict:
    """{relative path: sha256} of every file in the single Output tree."""
    output_roots = list(run_directory.glob("*/*/Output"))
    assert len(output_roots) == 1, output_roots
    return {
        str(path.relative_to(output_roots[0])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output_roots[0].rglob("*"))
        if path.is_file()
    }


def checkpoints_directory(run_directory: Path) -> Path:
    matches = list(run_directory.glob("*/*/Checkpoints"))
    assert len(matches) == 1, matches
    return matches[0]


class SimulatedCrash(RuntimeError):
    """Raised from the step callback to abort like a kill would."""


def crash_after(stop_step: int):
    def callback(simulation):
        if simulation.num_step >= stop_step:
            raise SimulatedCrash(f"simulated crash at step {stop_step}")
        return False

    return callback


AUTOSAVE_INTERVAL = 3
CRASH_STEP = 8


# --------------------------------------------------------------------------
# Autosave writer
# --------------------------------------------------------------------------


def test_autosave_off_by_default_writes_no_checkpoints(tmp_path):
    run_directory = prepare_run_directory(tmp_path, "no_autosave", end_time=1.0)
    simulation = run_simulation(run_directory)
    assert simulation.transient_input["AUTOSAVE_INTERVAL"] is None
    assert simulation.transient_input["RESTART"] is False
    assert list(run_directory.glob("*/*/Checkpoints")) == []


def test_autosave_writes_and_prunes_checkpoints(tmp_path):
    run_directory = prepare_run_directory(tmp_path, "autosave", end_time=1.0)
    set_simulation_keys(run_directory, autosave_interval=2)
    simulation = run_simulation(run_directory)
    checkpoint_files = sorted(
        path.name for path in checkpoints_directory(run_directory).iterdir()
    )
    # Pruned to the two most recent checkpoints, named by their step.
    assert len(checkpoint_files) == 2
    last_step = 2 * (simulation.num_step // 2)
    assert checkpoint_files == [
        f"checkpoint_step_{last_step - 2:08d}.npz",
        f"checkpoint_step_{last_step:08d}.npz",
    ]


# --------------------------------------------------------------------------
# Restart
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "network_section", [None, NETWORK_SECTION], ids=["plain", "coupled"]
)
def test_restart_resumes_bit_identically(tmp_path, network_section):
    straight_directory = prepare_run_directory(
        tmp_path, "straight", network_section=network_section
    )
    set_simulation_keys(straight_directory, autosave_interval=AUTOSAVE_INTERVAL)
    run_simulation(straight_directory)
    reference_hashes = output_tree_hashes(straight_directory)

    resumed_directory = prepare_run_directory(
        tmp_path, "resumed", network_section=network_section
    )
    set_simulation_keys(resumed_directory, autosave_interval=AUTOSAVE_INTERVAL)
    interrupted = Simulation(
        str(resumed_directory), step_callback=crash_after(CRASH_STEP)
    )
    with pytest.raises(SimulatedCrash):
        interrupted.run()
    assert interrupted.num_step == CRASH_STEP
    # The crash happened after at least one autosave.
    assert len(list(checkpoints_directory(resumed_directory).iterdir())) > 0

    set_simulation_keys(resumed_directory, restart=True)
    run_simulation(resumed_directory)

    resumed_hashes = output_tree_hashes(resumed_directory)
    differing = sorted(
        name
        for name in set(reference_hashes) | set(resumed_hashes)
        if reference_hashes.get(name) != resumed_hashes.get(name)
    )
    assert not differing, (
        "restarted run is not bit-identical to the straight run; differing "
        f"files: {differing}"
    )


def test_restart_without_checkpoint_errors(tmp_path):
    run_directory = prepare_run_directory(tmp_path, "no_checkpoint", end_time=1.0)
    set_simulation_keys(run_directory, restart=True)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        Simulation(str(run_directory)).run()


def test_restart_with_changed_method_errors_on_manifest(tmp_path):
    """All methods share the simulation_results/<name>/... subtree, so the
    checkpoint of the interrupted run is found and the restart of an edited
    deck fails with the manifest integration-method check (also exercised
    below at unit level)."""
    run_directory = prepare_run_directory(tmp_path, "method_mismatch", end_time=1.0)
    set_simulation_keys(run_directory, autosave_interval=2)
    interrupted = Simulation(
        str(run_directory), step_callback=crash_after(6)
    )
    with pytest.raises(SimulatedCrash):
        interrupted.run()

    conductor_file = run_directory / "conductor_CONDUCTOR_1.yaml"
    conductor_document = yaml.safe_load(conductor_file.read_text())
    conductor_document["conductor"]["inputs"]["thermohydraulic_method"] = "CN"
    conductor_file.write_text(
        yaml.safe_dump(conductor_document, sort_keys=False)
    )
    set_simulation_keys(run_directory, restart=True)
    with pytest.raises(ValueError, match="integration method"):
        Simulation(str(run_directory)).run()


def test_manifest_method_mismatch_errors_at_unit_level(tmp_path):
    """The manifest integration-method check itself (defense against a
    checkpoint written by a different configuration into the same tree)."""
    import io
    import types

    import numpy as np

    from conductor.conductor_flags import MethodFlag
    from utility_functions.checkpoint import _restore_conductor_state

    archive_buffer = io.BytesIO()
    np.savez(
        archive_buffer,
        **{
            "conductor/CONDUCTOR_1/solution": np.zeros((4, 2)),
            "conductor/CONDUCTOR_1/method": np.asarray(
                MethodFlag.CRANK_NICOLSON.value
            ),
        },
    )
    archive_buffer.seek(0)
    stub_conductor = types.SimpleNamespace(
        identifier="CONDUCTOR_1",
        inputs=types.SimpleNamespace(
            thermohydraulic_method=MethodFlag.BACKWARD_EULER
        ),
    )
    with np.load(archive_buffer) as archive:
        with pytest.raises(ValueError, match="integration method"):
            _restore_conductor_state(archive, stub_conductor)
