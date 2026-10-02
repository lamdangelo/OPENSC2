"""Tests of the end-of-run summary (simulation.log) and hot-spot log.

Unit tests exercise the tracker, the feature detection and the formatter on
synthetic stubs; the integration tests run the shortened CASE_1 scenario
(see test_network_coupling) once without and once with the hydraulic
network and check the produced files.
"""

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utility_functions.simulation_log import (
    ComponentExtrema,
    SimulationSummary,
    conductor_level_extremum,
    detect_features,
    format_log,
)
from electromagnetics.electromagnetic_flags import CurrentMode

from test_network_coupling import (
    NETWORK_SECTION,
    SHORTENED_END_TIME,
    prepare_run_directory,
    run_simulation,
)


# --------------------------------------------------------------------------
# Unit tests: extrema tracking
# --------------------------------------------------------------------------


COORDINATES = np.array([0.0, 1.0, 2.0])


def test_observe_max_records_value_time_location():
    extrema = ComponentExtrema("CHAN_1", "fluid")
    extrema.observe_max(
        "max_temperature", np.array([4.0, 6.0, 5.0]), 0.5, COORDINATES
    )
    record = extrema.records["max_temperature"]
    assert (record.value, record.time, record.location) == (6.0, 0.5, 1.0)


def test_observe_max_keeps_earlier_larger_value():
    extrema = ComponentExtrema("CHAN_1", "fluid")
    extrema.observe_max(
        "max_temperature", np.array([4.0, 6.0, 5.0]), 0.5, COORDINATES
    )
    extrema.observe_max(
        "max_temperature", np.array([5.0, 5.5, 5.9]), 1.0, COORDINATES
    )
    record = extrema.records["max_temperature"]
    assert (record.value, record.time) == (6.0, 0.5)
    extrema.observe_max(
        "max_temperature", np.array([4.0, 4.0, 7.0]), 1.5, COORDINATES
    )
    record = extrema.records["max_temperature"]
    assert (record.value, record.time, record.location) == (7.0, 1.5, 2.0)


def test_observe_min_with_negative_values():
    extrema = ComponentExtrema("CHAN_1", "fluid")
    extrema.observe_min(
        "min_velocity", np.array([0.5, -0.4, 0.1]), 0.2, COORDINATES
    )
    extrema.observe_min(
        "min_velocity", np.array([0.5, -0.1, 0.1]), 0.4, COORDINATES
    )
    record = extrema.records["min_velocity"]
    assert (record.value, record.time, record.location) == (-0.4, 0.2, 1.0)


def test_observe_max_ravels_column_shaped_arrays():
    extrema = ComponentExtrema("JACKET_1", "solid")
    extrema.observe_max(
        "max_temperature", np.array([[4.0], [8.0], [5.0]]), 1.0, COORDINATES
    )
    record = extrema.records["max_temperature"]
    assert (record.value, record.location) == (8.0, 1.0)


def test_observe_max_accepts_scalar_without_location():
    extrema = ComponentExtrema("STRAND_1", "solid")
    extrema.observe_max("max_abs_end_to_end_voltage", 2.5e-3, 1.0)
    record = extrema.records["max_abs_end_to_end_voltage"]
    assert (record.value, record.time, record.location) == (2.5e-3, 1.0, None)


def test_conductor_level_extremum_picks_winning_component():
    chan = ComponentExtrema("CHAN_1", "fluid")
    chan.observe_max("max_temperature", np.array([6.8]), 1.2, COORDINATES)
    strand = ComponentExtrema("STR_MIX_1", "solid")
    strand.observe_max("max_temperature", np.array([7.1]), 1.3, COORDINATES)
    winner = conductor_level_extremum([chan, strand], "max_temperature")
    assert winner[0] == "STR_MIX_1"
    assert winner[1].value == 7.1
    assert conductor_level_extremum([chan, strand], "max_abs_tap_voltage") is None


def test_conductor_level_extremum_minimum():
    chan_1 = ComponentExtrema("CHAN_1", "fluid")
    chan_1.observe_min("min_velocity", np.array([-0.2]), 1.0, COORDINATES)
    chan_2 = ComponentExtrema("CHAN_2", "fluid")
    chan_2.observe_min("min_velocity", np.array([-0.7]), 1.5, COORDINATES)
    winner = conductor_level_extremum([chan_1, chan_2], "min_velocity", minimum=True)
    assert winner[0] == "CHAN_2"
    assert winner[1].value == -0.7


# --------------------------------------------------------------------------
# Unit tests: feature detection and formatting (stub conductors)
# --------------------------------------------------------------------------


def make_operations(**overrides):
    operations = dict(
        coupling_loss_time_constant=0.0,
        eddy_loss_geometry_constant=0.0,
        filament_diameter=0.0,
        transverse_coupling_file="",
        magnetic_field_scales_with_current=False,
    )
    operations.update(overrides)
    return SimpleNamespace(**operations)


def make_collection(*components):
    return SimpleNamespace(collection=list(components), number=len(components))


def make_stub_conductor(strands=(), jackets=(), fluids=(), network_ports=()):
    return SimpleNamespace(
        identifier="CONDUCTOR_1",
        network_ports=list(network_ports),
        inventory=SimpleNamespace(
            strands=make_collection(*strands),
            jackets=make_collection(*jackets),
            fluids=make_collection(*fluids),
        ),
        inputs=SimpleNamespace(
            thermohydraulic_method=SimpleNamespace(name="BDF2"),
            current_mode=CurrentMode.CURRENT_NOT_DEFINED,
        ),
        hydraulic_formulation=SimpleNamespace(value="velocity"),
        mesh=SimpleNamespace(
            number_of_elements=50,
            number_of_nodes=51,
            node_coordinates=np.linspace(0.0, 2.0, 51),
        ),
    )


def test_detect_features_accepts_loop_family_lists_and_tape_flag():
    """Stacks declare several coupling loop families as lists and the
    thin-strip hysteresis through a boolean; the checklist must not compare
    a list with a float (regression: crash in the end-of-run summary)."""
    strand = SimpleNamespace(
        identifier="STACK_1",
        operations=make_operations(
            coupling_loss_time_constant=[0.983, 0.106],
            filament_diameter=0.0,
            tape_hysteresis_loss=True,
        ),
    )
    features = detect_features(make_stub_conductor(strands=[strand]))
    assert features["coupling loss"] is True
    assert features["hysteresis loss"] is True
    off = SimpleNamespace(
        identifier="STACK_2",
        operations=make_operations(coupling_loss_time_constant=[0.0, 0.0]),
    )
    assert detect_features(make_stub_conductor(strands=[off]))["coupling loss"] is False


def test_detect_features_all_disabled():
    strand = SimpleNamespace(identifier="STR_1", operations=make_operations())
    conductor = make_stub_conductor(strands=[strand])
    assert all(not active for active in detect_features(conductor).values())


@pytest.mark.parametrize(
    "override, feature",
    [
        (dict(coupling_loss_time_constant=58e-3), "coupling loss"),
        (dict(eddy_loss_geometry_constant=1.0), "eddy current loss"),
        (dict(filament_diameter=6e-6), "hysteresis loss"),
        (
            dict(transverse_coupling_file="coupling.tsv"),
            "transversal thermal coupling",
        ),
        (dict(magnetic_field_scales_with_current=True), "B ~ I scaling"),
    ],
)
def test_detect_features_flags(override, feature):
    strand = SimpleNamespace(
        identifier="STR_1", operations=make_operations(**override)
    )
    conductor = make_stub_conductor(strands=[strand])
    features = detect_features(conductor)
    assert features[feature] is True
    assert sum(features.values()) == 1


def test_detect_features_casing_bath_and_network():
    strand = SimpleNamespace(
        identifier="STR_1",
        operations=make_operations(),
        _transverse_patches=[{"partner": None}, {"partner": object()}],
    )
    conductor = make_stub_conductor(strands=[strand], network_ports=["port"])
    features = detect_features(conductor)
    assert features["casing bath (BOUNDARY sink)"] is True
    assert features["hydraulic network"] is True


def make_stub_simulation(conductor, restart=False):
    return SimpleNamespace(
        transient_input={
            "SIMULATION": "STUB_RUN",
            "MAGNET": "conductor_definition.yaml",
            "TEND": 2.0,
            "IADAPTIME": 0,
            "STPMIN": 1.0e-3,
            "STPMAX": 1.0e-3,
            "RESTART": restart,
        },
        num_step=10,
        simulation_time=[0.0, 2.0],
        list_of_Conductors=[conductor],
    )


TIMESTAMP_PATTERN = r"\d{2}\.\d{2}\.\d{2} : \d{2}:\d{2}:\d{2}"


def test_format_log_header_and_missing_voltage():
    strand = SimpleNamespace(identifier="STR_1", operations=make_operations())
    conductor = make_stub_conductor(strands=[strand])
    text = format_log(
        make_stub_simulation(conductor),
        {"CONDUCTOR_1": {}},
        cpu_seconds=1.0,
        wall_seconds=2.0,
    )
    assert re.search(rf"Simulation ended\s+: {TIMESTAMP_PATTERN}", text)
    assert "Model (SIMULATION)      : STUB_RUN" in text
    assert "Total time steps        : 10" in text
    assert "Integration scheme      : BDF2" in text
    assert "Hydraulic formulation   : velocity" in text
    assert "Mesh                    : 50 elements, 51 nodes" in text
    assert text.count("n/a (electric module disabled)") == 2
    assert "hydraulic network            : false" in text
    assert "NOTE: restarted run" not in text


def test_format_log_reports_extrema_with_component():
    strand = SimpleNamespace(identifier="STR_1", operations=make_operations())
    conductor = make_stub_conductor(strands=[strand])
    extrema = ComponentExtrema("STR_1", "solid")
    extrema.observe_max(
        "max_temperature", np.array([4.0, 7.1, 5.0]), 1.25, COORDINATES
    )
    text = format_log(
        make_stub_simulation(conductor, restart=True),
        {"CONDUCTOR_1": {"STR_1": extrema}},
        cpu_seconds=1.0,
        wall_seconds=2.0,
    )
    line = next(l for l in text.splitlines() if "max temperature" in l)
    assert "7.100000e+00 K" in line
    assert "at t = 1.2500 s, x = 1.0000 m" in line
    assert "(STR_1)" in line
    assert "NOTE: restarted run" in text


# --------------------------------------------------------------------------
# Unit tests: hot-spot buffering and flushing
# --------------------------------------------------------------------------


def make_tracked_conductor(time_now):
    node_fields = SimpleNamespace(
        temperature=np.array([4.5, 5.0, 4.7]),
        pressure=np.array([6.0e5, 5.9e5, 5.8e5]),
        velocity=np.array([0.1, 0.2, 0.15]),
        mass_flow_rate=np.array([1.0e-2, 1.1e-2, 1.05e-2]),
    )
    fluid = SimpleNamespace(
        identifier="CHAN_1", coolant=SimpleNamespace(node_fields=node_fields)
    )
    strand = SimpleNamespace(
        identifier="STR_1",
        operations=make_operations(),
        node_fields=SimpleNamespace(temperature=np.array([4.6, 5.2, 4.8])),
    )
    conductor = make_stub_conductor(strands=[strand], fluids=[fluid])
    conductor.cond_time = [time_now]
    return conductor


def make_hotspot_simulation(tmp_path, restart=False):
    return SimpleNamespace(
        dict_path={"Output_Time_evolution_CONDUCTOR_1_dir": str(tmp_path)},
        transient_input={"RESTART": restart},
    )


def test_hotspot_flush_appends_without_duplicate_header(tmp_path):
    simulation = make_hotspot_simulation(tmp_path)
    summary = SimulationSummary()
    for step, time_now in enumerate((0.0, 0.5, 1.0)):
        summary.update(simulation, make_tracked_conductor(time_now))
    summary.flush_hotspots(simulation)
    # A second flush cycle (e.g. after a restart) appends rows, not headers.
    summary.update(simulation, make_tracked_conductor(1.5))
    summary.flush_hotspots(simulation)

    hotspot = pd.read_csv(tmp_path / "hotspot_log.tsv", sep="\t")
    assert list(hotspot.columns) == [
        "time (s)",
        "CHAN_1_max_temperature",
        "STR_1_max_temperature",
    ]
    assert hotspot["time (s)"].tolist() == [0.0, 0.5, 1.0, 1.5]
    assert (hotspot["CHAN_1_max_temperature"] == 5.0).all()
    assert (hotspot["STR_1_max_temperature"] == 5.2).all()


def test_fresh_run_truncates_previous_hotspot_file(tmp_path):
    simulation = make_hotspot_simulation(tmp_path)
    stale = SimulationSummary()
    stale.update(simulation, make_tracked_conductor(0.0))
    stale.update(simulation, make_tracked_conductor(0.5))
    stale.flush_hotspots(simulation)

    fresh = SimulationSummary()
    fresh.update(simulation, make_tracked_conductor(0.0))
    fresh.flush_hotspots(simulation)
    # Mid-run flushes of the same run still append.
    fresh.update(simulation, make_tracked_conductor(0.5))
    fresh.flush_hotspots(simulation)

    hotspot = pd.read_csv(tmp_path / "hotspot_log.tsv", sep="\t")
    assert hotspot["time (s)"].tolist() == [0.0, 0.5]


def test_restarted_run_appends_to_truncated_hotspot_file(tmp_path):
    simulation = make_hotspot_simulation(tmp_path)
    interrupted = SimulationSummary()
    interrupted.update(simulation, make_tracked_conductor(0.0))
    interrupted.update(simulation, make_tracked_conductor(0.5))
    interrupted.flush_hotspots(simulation)

    restarted_simulation = make_hotspot_simulation(tmp_path, restart=True)
    resumed = SimulationSummary()
    resumed.update(restarted_simulation, make_tracked_conductor(1.0))
    resumed.flush_hotspots(restarted_simulation)

    hotspot = pd.read_csv(tmp_path / "hotspot_log.tsv", sep="\t")
    assert hotspot["time (s)"].tolist() == [0.0, 0.5, 1.0]


def test_update_tracks_both_voltage_definitions(tmp_path):
    simulation = make_hotspot_simulation(tmp_path)
    conductor = make_tracked_conductor(1.0)
    conductor.inputs.current_mode = CurrentMode.CURRENT_IS_CONSTANT
    strand = conductor.inventory.strands.collection[0]
    strand.gauss_fields = SimpleNamespace(
        has=lambda name: name == "delta_voltage_along_sum",
        delta_voltage_along_sum=np.array([0.0, -2.3e-3, 1.1e-3]),
    )
    # Two strand-interleaved potentials of a single strand: phi[0], phi[-1].
    conductor.nodal_potential = np.array([1.0e-3, 0.5e-3, -0.9e-3])

    summary = SimulationSummary()
    summary.update(simulation, conductor)
    records = summary._per_conductor["CONDUCTOR_1"]["STR_1"].records
    assert records["max_abs_tap_voltage"].value == pytest.approx(2.3e-3)
    assert records["max_abs_tap_voltage"].location is None
    assert records["max_abs_end_to_end_voltage"].value == pytest.approx(1.9e-3)


def test_update_record_hotspot_false_skips_row(tmp_path):
    simulation = make_hotspot_simulation(tmp_path)
    summary = SimulationSummary()
    summary.update(
        simulation, make_tracked_conductor(1.0), record_hotspot=False
    )
    summary.flush_hotspots(simulation)
    assert not (tmp_path / "hotspot_log.tsv").exists()
    # The extrema were still seeded.
    records = summary._per_conductor["CONDUCTOR_1"]["CHAN_1"].records
    assert records["max_temperature"].value == 5.0


# --------------------------------------------------------------------------
# Integration: shortened CASE_1 runs
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    run_directory = prepare_run_directory(
        tmp_path_factory.mktemp("simulation_log"), "plain"
    )
    return run_directory, run_simulation(run_directory)


@pytest.fixture(scope="module")
def network_run(tmp_path_factory):
    run_directory = prepare_run_directory(
        tmp_path_factory.mktemp("simulation_log_network"),
        "network",
        network_section=NETWORK_SECTION,
    )
    return run_directory, run_simulation(run_directory)


def read_simulation_log(run_directory):
    log_paths = list(run_directory.glob("*/*/simulation.log"))
    assert len(log_paths) == 1, log_paths
    return log_paths[0].read_text()


def test_simulation_log_written_with_metadata(plain_run):
    run_directory, simulation = plain_run
    text = read_simulation_log(run_directory)
    assert re.search(rf"Simulation ended\s+: {TIMESTAMP_PATTERN}", text)
    assert f"Total time steps        : {simulation.num_step}" in text
    assert "Hydraulic formulation   : velocity" in text
    assert "hydraulic network            : false" in text
    conductor = simulation.list_of_Conductors[0]
    assert (
        f"Mesh                    : {conductor.mesh.number_of_elements} elements,"
        f" {conductor.mesh.number_of_nodes} nodes"
    ) in text
    assert f"requested TEND = {SHORTENED_END_TIME:.6g} s" in text


def test_simulation_log_times_and_locations_in_range(plain_run):
    run_directory, simulation = plain_run
    text = read_simulation_log(run_directory)
    conductor = simulation.list_of_Conductors[0]
    length = conductor.mesh.node_coordinates[-1]
    times = [float(t) for t in re.findall(r"at t = (\S+) s", text)]
    locations = [float(x) for x in re.findall(r"x = (\S+) m", text)]
    assert times and locations
    assert all(0.0 <= t <= SHORTENED_END_TIME + 1e-9 for t in times)
    assert all(0.0 <= x <= length + 1e-9 for x in locations)


def test_hotspot_log_written_per_step(plain_run):
    run_directory, simulation = plain_run
    conductor = simulation.list_of_Conductors[0]
    hotspot_paths = list(
        run_directory.glob(
            f"*/*/Output/Time_evolution/{conductor.identifier}/hotspot_log.tsv"
        )
    )
    assert len(hotspot_paths) == 1, hotspot_paths
    hotspot = pd.read_csv(hotspot_paths[0], sep="\t")
    expected_columns = 1 + (
        conductor.inventory.fluids.number
        + conductor.inventory.strands.number
        + conductor.inventory.jackets.number
    )
    assert hotspot.shape[1] == expected_columns
    assert hotspot.columns[0] == "time (s)"
    assert len(hotspot) == simulation.num_step + 1
    assert (np.diff(hotspot["time (s)"].to_numpy()) > 0.0).all()
    temperatures = hotspot.iloc[:, 1:].to_numpy()
    assert (temperatures > 3.0).all()


def test_simulation_log_network_run(network_run):
    run_directory, simulation = network_run
    text = read_simulation_log(run_directory)
    assert "Hydraulic formulation   : mass_flow" in text
    assert "hydraulic network            : true" in text


# --------------------------------------------------------------------------
# Unit tests: normal-zone length and NZPV estimate
# --------------------------------------------------------------------------


def test_normal_zone_length_from_regime_data():
    from utility_functions.simulation_log import normal_zone_length

    strand = SimpleNamespace(
        _electric_regime_gauss={
            "sharing": np.array([2, 3]),
            "normal": np.array([4]),
        }
    )
    lengths = np.full(6, 0.5)
    assert normal_zone_length(strand, lengths) == pytest.approx(1.5)


def test_normal_zone_length_none_without_regime_data():
    from utility_functions.simulation_log import normal_zone_length

    assert normal_zone_length(SimpleNamespace(), np.ones(4)) is None


def test_normal_zone_length_zero_when_superconducting():
    from utility_functions.simulation_log import normal_zone_length

    strand = SimpleNamespace(
        _electric_regime_gauss={
            "sharing": np.empty(0, dtype=int),
            "normal": np.empty(0, dtype=int),
        }
    )
    assert normal_zone_length(strand, np.ones(4)) == 0.0


def test_estimate_nzpv_linear_growth():
    from utility_functions.simulation_log import (
        estimate_normal_zone_propagation,
    )

    times = np.linspace(0.0, 1.0, 101)
    # Zone appears at t = 0.1 and grows at 4 m/s (2 m/s per front).
    lengths = np.clip(4.0 * (times - 0.1), 0.0, None)
    estimate = estimate_normal_zone_propagation(times, lengths)
    assert estimate is not None
    assert estimate["onset_time"] == pytest.approx(0.11, abs=0.02)
    assert estimate["max_length"] == pytest.approx(3.6)
    assert estimate["velocity"] == pytest.approx(2.0, rel=0.05)


def test_estimate_nzpv_no_zone():
    from utility_functions.simulation_log import (
        estimate_normal_zone_propagation,
    )

    assert (
        estimate_normal_zone_propagation([0.0, 1.0], [0.0, 0.0]) is None
    )


def test_estimate_nzpv_degenerate_window_has_no_velocity():
    from utility_functions.simulation_log import (
        estimate_normal_zone_propagation,
    )

    # Heater-made zone of constant length: onset reported, no velocity.
    estimate = estimate_normal_zone_propagation(
        [0.0, 0.5, 1.0], [0.1, 0.1, 0.1]
    )
    assert estimate is not None
    assert estimate["velocity"] is None


def test_format_log_normal_zone_section():
    strand = SimpleNamespace(identifier="STACK1_V1", operations=make_operations())
    conductor = make_stub_conductor(strands=[strand])
    times = list(np.linspace(0.0, 1.0, 101))
    lengths = list(np.clip(4.0 * (np.asarray(times) - 0.1), 0.0, None))
    text = format_log(
        make_stub_simulation(conductor),
        {"CONDUCTOR_1": {}},
        cpu_seconds=1.0,
        wall_seconds=2.0,
        normal_zone={
            "CONDUCTOR_1": {
                "STACK1_V1": {"time": times, "length": lengths}
            }
        },
    )
    assert "Normal zone" in text
    assert "STACK1_V1" in text
    assert "NZPV ~ " in text
    assert "m/s per front" in text


def test_estimate_nzpv_ignition_only_is_not_propagation():
    from utility_functions.simulation_log import (
        estimate_normal_zone_propagation,
    )

    # Heater ignites 0.5 m within 2 ms, zone then holds and collapses:
    # the pre-disturbance ignition ramp must NOT be measured as NZPV.
    times = [0.0, 0.001, 0.002, 0.04, 0.1, 0.5, 1.0]
    lengths = [0.0, 0.25, 0.5, 0.5, 0.5, 0.2, 0.0]
    estimate = estimate_normal_zone_propagation(
        times, lengths, disturbance_end=0.04
    )
    assert estimate is not None
    assert estimate["velocity"] is None
    assert estimate["max_length"] == pytest.approx(0.5)


def test_estimate_nzpv_growth_after_disturbance():
    from utility_functions.simulation_log import (
        estimate_normal_zone_propagation,
    )

    # 0.5 m ignited by the heater, then true propagation at 2 m/s total
    # (1 m/s per front) after heater-off at t = 0.04 s.
    times = np.linspace(0.0, 2.0, 401)
    lengths = np.where(
        times < 0.002, times * 250.0,
        np.where(times < 0.04, 0.5, 0.5 + 2.0 * (times - 0.04)),
    )
    estimate = estimate_normal_zone_propagation(
        times, lengths, disturbance_end=0.04
    )
    assert estimate["velocity"] == pytest.approx(1.0, rel=0.05)
