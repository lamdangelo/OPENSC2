"""Unit tests of the nonlocal transverse-conduction heat source.

The file-driven coupling exchanges q = g * (T_partner(x_p) - T_local(x))
between winding-geometry-adjacent positions. The tests drive bare
SolidComponents with prescribed temperature fields and check the closed
form, the orientation reversal (decreasing partner interval), the
accumulation of overlapping patches, the pairwise antisymmetry (zero
net energy) and the opt-in behaviour (no file -> exactly zero power).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from opensc2.adapters.input.external_inputs import FileExternalInputReader
from opensc2.application.build_simulation import wire_transverse_coupling
from opensc2.components.solid.solid_component import SolidComponent
from factories import build_settings, make_conductor, make_jacket, solid_operations

NUMBER_OF_NODES = 11
LENGTH = 10.0
CONDUCTANCE = 0.7  # W/(m K)


def make_reader(tmp_path) -> FileExternalInputReader:
    """Reader resolving relative file names against the deck directory."""
    return FileExternalInputReader(
        SimpleNamespace(structure_elements_path=tmp_path / "conductor.yaml")
    )


def make_component(identifier: str, coupling_file: str,
                   temperature: np.ndarray, tmp_path) -> SolidComponent:
    component = make_jacket(
        identifier=identifier,
        operations=solid_operations(transverse_coupling_file=coupling_file),
        build=build_settings(external_inputs=make_reader(tmp_path)),
    )
    component.node_fields.temperature = temperature
    return component


def make_world(tmp_path, files: dict, temperatures: dict):
    """Two single-component conductors CONDUCTOR_A / CONDUCTOR_B, wired."""
    conductors = []
    components = {}
    for name in ("A", "B"):
        component = make_component(
            f"COMP_{name}", files.get(name, ""), temperatures[name], tmp_path
        )
        conductor = make_conductor(
            number_of_nodes=NUMBER_OF_NODES,
            length=LENGTH,
            identifier=f"CONDUCTOR_{name}",
            components=[component],
        )
        conductors.append(conductor)
        components[name] = component
    wire_transverse_coupling(conductors)
    simulation = SimpleNamespace(list_of_Conductors=conductors)
    return conductors, components, simulation


def write_patch_file(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(",".join(str(v) for v in row) for row in rows))
    return name


def test_matches_closed_form(tmp_path):
    """Uniform-gradient partner: q = g*(T_B(x) - T_A(x)) node by node."""
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, LENGTH, "CONDUCTOR_B", "COMP_B", 0.0, LENGTH, CONDUCTANCE]],
    )
    _, comps, sim = make_world(
        tmp_path, {"A": file_a},
        {"A": np.full(NUMBER_OF_NODES, 6.0), "B": 6.0 + 0.5 * x},
    )
    conductor_a = sim.list_of_Conductors[0]
    comps["A"].get_transverse_coupling(conductor_a)  # allocates only
    comps["A"].get_transverse_coupling(conductor_a)
    power = comps["A"].node_fields.transverse_coupling_linear_power[:, 0]
    assert np.allclose(power, CONDUCTANCE * 0.5 * x)


def test_reversed_interval_maps_mirror(tmp_path):
    """A decreasing partner interval must sample the partner backwards."""
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, LENGTH, "CONDUCTOR_B", "COMP_B", LENGTH, 0.0, CONDUCTANCE]],
    )
    _, comps, sim = make_world(
        tmp_path, {"A": file_a},
        {"A": np.full(NUMBER_OF_NODES, 6.0), "B": 6.0 + 0.5 * x},
    )
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    power = comps["A"].node_fields.transverse_coupling_linear_power[:, 0]
    assert np.allclose(power, CONDUCTANCE * 0.5 * (LENGTH - x))


def test_overlapping_patches_accumulate(tmp_path):
    """Two identical patches must give exactly twice the exchange."""
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    row = [0.0, LENGTH, "CONDUCTOR_B", "COMP_B", 0.0, LENGTH, CONDUCTANCE]
    file_a = write_patch_file(tmp_path, "a.csv", [row, row])
    _, comps, sim = make_world(
        tmp_path, {"A": file_a},
        {"A": np.full(NUMBER_OF_NODES, 6.0), "B": 6.0 + 0.5 * x},
    )
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    power = comps["A"].node_fields.transverse_coupling_linear_power[:, 0]
    assert np.allclose(power, 2.0 * CONDUCTANCE * 0.5 * x)


def test_pair_is_antisymmetric(tmp_path):
    """Both sides listed with the same conductance: net power is zero.

    Matching grids and a linear partner profile make the interpolation
    exact, so the antisymmetry must hold to machine precision node by
    node (the pair map here is the identity).
    """
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, LENGTH, "CONDUCTOR_B", "COMP_B", 0.0, LENGTH, CONDUCTANCE]],
    )
    file_b = write_patch_file(
        tmp_path, "b.csv",
        [[0.0, LENGTH, "CONDUCTOR_A", "COMP_A", 0.0, LENGTH, CONDUCTANCE]],
    )
    _, comps, sim = make_world(
        tmp_path, {"A": file_a, "B": file_b},
        {"A": 6.0 + 0.2 * x, "B": 7.0 - 0.3 * x},
    )
    for _ in range(2):
        comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
        comps["B"].get_transverse_coupling(sim.list_of_Conductors[1])
    total = (
        comps["A"].node_fields.transverse_coupling_linear_power[:, 0]
        + comps["B"].node_fields.transverse_coupling_linear_power[:, 0]
    )
    assert np.allclose(total, 0.0, atol=1e-14)


def test_no_file_gives_zero(tmp_path):
    """Opt-in: without a coupling file the power stays exactly zero."""
    _, comps, sim = make_world(
        tmp_path, {},
        {"A": np.full(NUMBER_OF_NODES, 6.0),
         "B": np.full(NUMBER_OF_NODES, 9.0)},
    )
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    assert np.all(
        comps["A"].node_fields.transverse_coupling_linear_power == 0.0
    )


def test_temperature_dependent_conductance(tmp_path):
    """Optional columns: g_eff = g * (min(T_mean, T_sat)/T_ref)^n."""
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    exponent, t_ref, t_sat = 1.5, 10.0, 12.0
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, LENGTH, "CONDUCTOR_B", "COMP_B", 0.0, LENGTH, CONDUCTANCE,
          exponent, t_ref, t_sat]],
    )
    temperature_a = np.full(NUMBER_OF_NODES, 6.0)
    temperature_b = 6.0 + 2.0 * x  # mean crosses the saturation cap
    _, comps, sim = make_world(
        tmp_path, {"A": file_a},
        {"A": temperature_a, "B": temperature_b},
    )
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    comps["A"].get_transverse_coupling(sim.list_of_Conductors[0])
    power = comps["A"].node_fields.transverse_coupling_linear_power[:, 0]
    mean = np.minimum(0.5 * (temperature_a + temperature_b), t_sat)
    expected = (
        CONDUCTANCE * (mean / t_ref) ** exponent
        * (temperature_b - temperature_a)
    )
    assert np.allclose(power, expected)


# --------------------------------------------------------------------------
# Reader and wiring
# --------------------------------------------------------------------------

def test_reader_parses_rows_defaults_comments_and_boundary(tmp_path):
    name = write_patch_file(
        tmp_path, "rows.csv",
        [
            ["# x_low, x_high, partner, component, xs, xe, g, n, T_ref, T_sat"],
            [0.0, 2.0, "CONDUCTOR_B", "COMP_B", 2.0, 0.0, 0.7],
            [2.0, 4.0, "CONDUCTOR_B", "COMP_B", 0.0, 2.0, 0.5, 1.5, 10.0, 12.0],
            [4.0, 6.0, "BOUNDARY", 4.5, 0.0, 0.0, 0.1],
        ],
    )
    rows = make_reader(tmp_path).transverse_coupling_rows(name)
    assert len(rows) == 3
    first, second, third = rows
    assert (first.x_low, first.x_high, first.partner_x_start, first.partner_x_end) == (0.0, 2.0, 2.0, 0.0)
    assert first.conductance == 0.7
    assert (first.temperature_exponent, first.reference_temperature) == (0.0, 1.0)
    assert first.saturation_temperature == np.inf
    assert not first.is_boundary
    assert (second.temperature_exponent, second.reference_temperature, second.saturation_temperature) == (1.5, 10.0, 12.0)
    assert third.is_boundary and third.boundary_temperature == 4.5


def test_rows_are_read_at_construction_and_mapped_by_the_wiring(tmp_path):
    x = np.linspace(0.0, LENGTH, NUMBER_OF_NODES)
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, 4.0, "CONDUCTOR_B", "COMP_B", 0.0, 4.0, CONDUCTANCE],
         [20.0, 30.0, "CONDUCTOR_B", "COMP_B", 0.0, 1.0, CONDUCTANCE],  # no local node
         [8.0, LENGTH, "BOUNDARY", 4.5, 0.0, 0.0, 0.1]],
    )
    conductors, comps, _ = make_world(
        tmp_path, {"A": file_a},
        {"A": np.full(NUMBER_OF_NODES, 6.0), "B": 6.0 + 0.5 * x},
    )
    assert len(comps["A"].transverse_coupling_rows) == 3
    patches = comps["A"].transverse_patches
    assert len(patches) == 2  # the row covering no local node is dropped
    np.testing.assert_array_equal(patches[0]["indices"], np.flatnonzero(x <= 4.0))
    assert patches[0]["partner"]["component"] is comps["B"]
    assert patches[1]["partner"] is None and patches[1]["boundary_temperature"] == 4.5
    assert comps["B"].transverse_coupling_rows == () and comps["B"].transverse_patches == []


def test_unknown_partner_is_reported_by_the_wiring(tmp_path):
    file_a = write_patch_file(
        tmp_path, "a.csv",
        [[0.0, LENGTH, "CONDUCTOR_Z", "COMP_B", 0.0, LENGTH, CONDUCTANCE]],
    )
    with pytest.raises(KeyError, match="unknown partner CONDUCTOR_Z/COMP_B"):
        make_world(
            tmp_path, {"A": file_a},
            {"A": np.full(NUMBER_OF_NODES, 6.0), "B": np.full(NUMBER_OF_NODES, 6.0)},
        )
