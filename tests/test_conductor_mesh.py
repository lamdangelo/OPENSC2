"""Tests of ConductorMesh, in particular the external mesh file path
(``mesh_type: -1`` + ``grid.mesh_file``) and the HELIAS-VIPER refined-deck
generator built on it."""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import interfaces.coolprop_interface as cpi
from conductor.conductor_mesh import ConductorMesh, MeshType

sys.path.insert(0, str(Path(__file__).parent / "verification"))
from case_builders import (  # noqa: E402
    VERIFICATION_FLUID,
    ChannelProbe,
    StepDriver,
    run_case,
    write_channel_run_directory,
)


@pytest.fixture
def constant_fluid():
    previous = cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    yield VERIFICATION_FLUID
    cpi.set_constant_fluid_properties(previous)

REPO = Path(__file__).resolve().parents[1]
HELIAS = REPO / "models" / "HELIAS-VIPER"

LENGTH = 10.0


def uniform_grid(number_of_elements: int) -> dict:
    return {
        "NELEMS": number_of_elements, "ITYMSH": 0, "NELREF": 0, "XBREFI": 0.0,
        "XEREFI": 0.0, "SIZMIN": 0.001, "SIZMAX": 2.0, "DXINCRE_LEFT": 1.2,
        "DXINCRE_RIGHT": 1.2, "MAXNOD": 10001,
    }


def write_mesh(path: Path, coordinates, header: str = "", fmt: str = "%.17g") -> Path:
    lines = [header] if header else []
    lines.extend(fmt % value for value in coordinates)
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Parsing and synchronisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("header", ["", "z [m]", "# z [m]"])
def test_from_file_reads_plain_and_headed_files(tmp_path, header):
    coordinates = np.array([0.0, 0.5, 1.5, 4.0, 10.0])
    path = write_mesh(tmp_path / "mesh.csv", coordinates, header)
    mesh = ConductorMesh.from_file(LENGTH, path, "C1")
    assert mesh.mesh_type is MeshType.FROM_FILE
    assert mesh.mesh_file == path
    np.testing.assert_array_equal(mesh.node_coordinates, coordinates)
    assert mesh.number_of_nodes == 5
    assert mesh.number_of_elements == 4


def test_from_file_ignores_blank_and_comment_lines(tmp_path):
    path = tmp_path / "mesh.csv"
    path.write_text("# comment\n\n0.0\n\n# another\n5.0\n10.0\n")
    mesh = ConductorMesh.from_file(LENGTH, path)
    np.testing.assert_array_equal(mesh.node_coordinates, [0.0, 5.0, 10.0])


def test_nelems_mismatch_warns_and_file_wins(tmp_path, caplog):
    path = write_mesh(tmp_path / "mesh.csv", np.linspace(0.0, LENGTH, 6))
    grid = {"ITYMSH": -1, "NELEMS": 99}
    with caplog.at_level(logging.WARNING, logger="opensc2Logger.discretization"):
        mesh = ConductorMesh(LENGTH, grid, path, "C1")
    assert mesh.number_of_elements == 5
    assert any("NELEMS = 99" in record.message for record in caplog.records)


def test_from_file_via_grid_dict_with_relative_key_semantics(tmp_path):
    """The loader passes the grid dict untouched plus the resolved path;
    the refinement keys may be missing entirely."""
    path = write_mesh(tmp_path / "mesh.csv", np.linspace(0.0, LENGTH, 3))
    mesh = ConductorMesh(LENGTH, {"ITYMSH": -1, "MESH_FILE": "mesh.csv"}, path)
    assert mesh.number_of_elements == 2
    assert np.isnan(mesh.maximum_number_of_nodes)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "coordinates, match",
    [
        ([0.0, 5.0, 4.0, 10.0], "strictly increasing"),
        ([0.0, 5.0, 5.0, 10.0], "strictly increasing"),
        ([0.1, 5.0, 10.0], "z0 must be 0.0"),
        ([0.0, 5.0, 9.0], "does not equal conductor_length"),
        ([0.0, float("nan"), 10.0], "non-finite"),
        ([0.0], "at least two"),
    ],
)
def test_from_file_rejects_invalid_coordinates(tmp_path, coordinates, match):
    path = write_mesh(tmp_path / "mesh.csv", coordinates)
    with pytest.raises(ValueError, match=match):
        ConductorMesh.from_file(LENGTH, path)


def test_from_file_rejects_non_numeric_body_line(tmp_path):
    path = tmp_path / "mesh.csv"
    path.write_text("0.0\nabc\n10.0\n")
    with pytest.raises(ValueError, match="line 2"):
        ConductorMesh.from_file(LENGTH, path)


def test_from_file_checks_maximum_number_of_nodes(tmp_path):
    path = write_mesh(tmp_path / "mesh.csv", np.linspace(0.0, LENGTH, 11))
    with pytest.raises(ValueError, match="should not exceed the maximum"):
        ConductorMesh.from_file(LENGTH, path, maximum_number_of_nodes=10)
    ConductorMesh.from_file(LENGTH, path, maximum_number_of_nodes=11)


def test_from_file_requires_path_and_existing_file(tmp_path):
    with pytest.raises(ValueError, match="mesh_file"):
        ConductorMesh(LENGTH, {"ITYMSH": -1})
    with pytest.raises(FileNotFoundError):
        ConductorMesh.from_file(LENGTH, tmp_path / "missing.csv")


def test_end_nodes_are_snapped_within_tolerance(tmp_path):
    path = write_mesh(tmp_path / "mesh.csv", [1e-7, 5.0, LENGTH + 1e-7])
    mesh = ConductorMesh.from_file(LENGTH, path)
    assert mesh.node_coordinates[0] == 0.0
    assert mesh.node_coordinates[-1] == LENGTH


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------

def test_derived_features_of_non_uniform_mesh(tmp_path):
    coordinates = np.array([0.0, 1.0, 1.5, 4.5, 10.0])
    path = write_mesh(tmp_path / "mesh.csv", coordinates)
    mesh = ConductorMesh.from_file(LENGTH, path)
    np.testing.assert_allclose(mesh.element_lengths, [1.0, 0.5, 3.0, 5.5])
    np.testing.assert_allclose(mesh.gauss_point_coordinates, [0.5, 1.25, 3.0, 7.25])
    np.testing.assert_allclose(mesh.dual_element_lengths, [0.5, 0.75, 1.75, 4.25, 2.75])
    assert mesh.min_element_length == 0.5
    assert mesh.max_element_length == 5.5


def test_uniform_file_matches_mesh_type_0(tmp_path):
    reference = ConductorMesh(LENGTH, uniform_grid(20))
    path = write_mesh(tmp_path / "mesh.csv", reference.node_coordinates)
    mesh = ConductorMesh.from_file(LENGTH, path)
    np.testing.assert_array_equal(mesh.node_coordinates, reference.node_coordinates)
    np.testing.assert_array_equal(mesh.element_lengths, reference.element_lengths)
    np.testing.assert_array_equal(mesh.dual_element_lengths, reference.dual_element_lengths)
    np.testing.assert_array_equal(mesh.gauss_point_coordinates, reference.gauss_point_coordinates)
    assert mesh.position_precision == reference.position_precision


# --------------------------------------------------------------------------
# End-to-end: a uniform mesh from file reproduces mesh_type 0 bit for bit
# --------------------------------------------------------------------------

def _run_channel_case(directory: Path, mesh_from_file: bool):
    run_directory = write_channel_run_directory(
        directory, number_of_elements=20, time_step=0.1, end_time=2.0,
        method="BDF2", hydraulic_boundary_condition=1,
        inlet_pressure=1.0e5 + 200.0, outlet_pressure=1.0e5,
        initial_pressure=1.0e5 + 100.0,
    )
    if mesh_from_file:
        conductor_path = run_directory / "conductor_CONDUCTOR_1.yaml"
        document = yaml.safe_load(conductor_path.read_text())
        length = float(document["conductor"]["inputs"]["length"])
        write_mesh(run_directory / "mesh.csv", np.linspace(0.0, length, 21))
        document["grid"]["mesh_type"] = -1
        document["grid"]["mesh_file"] = "mesh.csv"
        conductor_path.write_text(yaml.safe_dump(document, sort_keys=False))
    probe = ChannelProbe(node_index=10)
    simulation = run_case(run_directory, step_callback=StepDriver(probe))
    probe.record_final(simulation)
    conductor = simulation.list_of_Conductors[0]
    channel = conductor.inventory.fluids.collection[0]
    return (
        conductor.mesh,
        probe.as_arrays(),
        np.array(channel.coolant.node_fields.pressure, dtype=float),
        np.array(channel.coolant.node_fields.temperature, dtype=float),
        channel.coordinate["z"].copy(),
    )


def test_uniform_mesh_file_run_is_bit_identical(constant_fluid, tmp_path):
    reference = _run_channel_case(tmp_path / "uniform", mesh_from_file=False)
    from_file = _run_channel_case(tmp_path / "from_file", mesh_from_file=True)
    assert from_file[0].mesh_type is MeshType.FROM_FILE
    np.testing.assert_array_equal(from_file[0].node_coordinates, reference[0].node_coordinates)
    for series_a, series_b in zip(reference[1], from_file[1]):
        np.testing.assert_array_equal(series_a, series_b)
    np.testing.assert_array_equal(reference[2], from_file[2])
    np.testing.assert_array_equal(reference[3], from_file[3])
    np.testing.assert_array_equal(reference[4], from_file[4])


# --------------------------------------------------------------------------
# HELIAS-VIPER refined-deck generator
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refine():
    sys.path.insert(0, str(HELIAS))
    import refine_2d_deck
    return refine_2d_deck


@pytest.mark.parametrize("zones", [
    [(0.0, 34.2), (34.2, 68.4), (923.7, 957.87)],
    [(0.0, 34.2), (1094.4, 1128.6), (1128.6, 1162.85)],
    [(100.0, 134.0)],
    [(0.0, 5.0), (7.0, 9.0), (950.0, 957.87)],
])
def test_graded_mesh_invariants(refine, zones):
    length = 957.87 if zones[-1][1] < 1000 else 1162.85
    fine, coarse, ratio = 0.01, 0.5, 1.2
    z = refine.build_graded_mesh(length, zones, fine, coarse, ratio)
    steps = np.diff(z)
    assert z[0] == 0.0 and z[-1] == length
    assert np.all(steps > 0.0)
    assert steps.max() <= coarse * 1.0001
    for start, end in zones:
        inside = (z >= start - 1e-9) & (z <= end + 1e-9)
        assert np.allclose(np.diff(z[inside]), fine, rtol=1e-6)
    # The geometric growth bound holds wherever a gap admits a full
    # transition (~3.25 m per side for 0.01 -> 0.5 m at 1.2); a gap that is
    # too short for that is stretched and may grow slightly faster.
    edges = [0.0] + [edge for zone in zones for edge in zone] + [length]
    gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges), 2)]
    if min(gap for gap in gaps if gap > 1e-9) > 10.0:
        growth = steps[1:] / steps[:-1]
        assert growth.max() <= ratio * 1.001 + 1e-9
        assert growth.min() >= 1.0 / (ratio * 1.001)


@pytest.mark.parametrize("number, expected", [
    (1, {0, 1, 27}), (2, {0, 32, 33}), (3, {0, 1, 31}),
])
def test_zone_turn_indices_of_baseline_decks(refine, number, expected):
    deck = HELIAS / f"2d_model_dp{number}"
    if not deck.is_dir():
        pytest.skip("baseline deck not present")
    design_point = refine.DESIGN_POINTS[number]
    assert refine.zone_turn_indices(deck, design_point) == expected
    x_low, x_high = refine.disturbance_interval(deck)
    assert x_high - x_low == pytest.approx(0.1, abs=1e-6)
