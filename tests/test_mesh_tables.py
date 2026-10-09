"""The mesh and electric-topology tables of a conductor (nodal coordinates,
connectivity, node distances, strand contact nodes) are plain numpy arrays.

These tests pin the arrays to the pandas implementation they replaced (the
legacy code is re-implemented here as the oracle; comparisons are bitwise)
and exercise the inductance paths that consume them, including the
analytical path that failed with a KeyError on pandas >= 3 (positional
indexing of a label-indexed Series).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from components.solid.strand_component import StrandComponent
from conductor import conductor_mesh
from conductor.conductor import Conductor
from electromagnetics import circuit_topology, conductance, inductance
from electromagnetics.electromagnetic_flags import InductanceMode, SelfInductanceMode
from factories import make_strand_mixed


# --------------------------------------------------------------------------
# Toy conductor: one channel, helical strands, one jacket
# --------------------------------------------------------------------------

def _collection(items):
    return SimpleNamespace(collection=list(items), number=len(items))


def _component(identifier, coordinate, strand=False):
    if strand:
        obj = make_strand_mixed(identifier=identifier, cos_theta=0.95)
        obj.coordinate = coordinate
        return obj
    return SimpleNamespace(identifier=identifier, coordinate=coordinate)


def make_conductor(n_elements=6, n_strands=2, length=0.3, pitch=0.4, helix_radius=0.01):
    z = np.linspace(0.0, length, n_elements + 1)
    axis = dict(x=np.zeros_like(z), y=np.zeros_like(z), z=z.copy())
    fluids = [_component("CHAN_1", axis)]
    strands = []
    for k in range(n_strands):
        theta = 2.0 * np.pi * z / pitch + 2.0 * np.pi * k / n_strands
        strands.append(
            _component(
                f"STR_{k + 1}",
                dict(x=helix_radius * np.cos(theta), y=helix_radius * np.sin(theta), z=z.copy()),
                strand=True,
            )
        )
    jackets = [_component("Z_JACKET_1", dict(axis))]
    components = fluids + strands + jackets
    n_all = len(components)
    n_cc = n_elements * n_strands

    # every strand pair in contact (flag matrix: row/col 0 is the environment)
    flag = np.zeros((1 + n_all, 1 + n_all))
    base = 1 + len(fluids)
    for a in range(n_strands):
        for b in range(n_strands):
            if a != b:
                flag[base + a, base + b] = 1.0

    return SimpleNamespace(
        identifier="COND",
        inventory=SimpleNamespace(
            fluids=_collection(fluids),
            strands=_collection(strands),
            jackets=_collection(jackets),
            all_components=_collection(components),
        ),
        mesh=SimpleNamespace(number_of_elements=n_elements),
        total_nodes=(n_elements + 1) * n_all,
        total_elements=n_elements * n_all,
        total_nodes_current_carriers=(n_elements + 1) * n_strands,
        electric=SimpleNamespace(
            total_elements_current_carriers=n_cc,
            inductance_matrix=np.zeros((n_cc, n_cc)),
        ),
        operations=SimpleNamespace(
            inductance_mode=InductanceMode.ANALYTICAL,
            self_inductance_mode=SelfInductanceMode.MODE_2,
        ),
        coupling=SimpleNamespace(contact_perimeter_flag=SimpleNamespace(matrix=flag)),
    )


def build_tables(conductor):
    """Same call sequence as Conductor.__build_electric_topology."""
    conductor_mesh.initialize_mesh_tables(conductor)
    nn = 0
    inventory = conductor.inventory
    for collection in (inventory.fluids, inventory.strands, inventory.jackets):
        conductor_mesh.build_nodal_coordinates(conductor, nn, collection)
        conductor_mesh.build_connectivity(conductor, nn, collection)
        nn += collection.number
    circuit_topology.build_connectivity_current_carriers(conductor)
    conductor_mesh.compute_node_distance(conductor)
    conductor_mesh.compute_gauss_node_distance(conductor)
    circuit_topology.detect_contacts(conductor)
    return conductor


# --------------------------------------------------------------------------
# Legacy pandas implementation (oracle)
# --------------------------------------------------------------------------

def legacy_tables(conductor):
    inventory = conductor.inventory
    n_all = inventory.all_components.number
    n_cs = conductor.mesh.number_of_elements + 1
    identifiers = [obj.identifier for obj in inventory.all_components.collection]
    kinds = []
    for obj in inventory.all_components.collection:
        if isinstance(obj, StrandComponent):
            kinds.append("StrandComponent")
        elif obj in inventory.fluids.collection:
            kinds.append("FluidComponent")
        else:
            kinds.append("JacketComponent")
    index = pd.MultiIndex.from_arrays(
        [
            pd.CategoricalIndex(
                np.tile(kinds, n_cs), ordered=True,
                categories=["FluidComponent", "StrandComponent", "JacketComponent"],
            ),
            pd.CategoricalIndex(np.tile(identifiers, n_cs), ordered=True, categories=identifiers),
        ],
        names=["Kind", "Identifier"],
    )
    nc = pd.DataFrame(
        dict(x=np.zeros(conductor.total_nodes), y=np.zeros(conductor.total_nodes), z=np.zeros(conductor.total_nodes)),
        index=index,
    )
    cm = pd.DataFrame(
        dict(start=np.zeros(conductor.total_elements, dtype=int), end=np.zeros(conductor.total_elements, dtype=int)),
        index=index[:-n_all],
    )
    for ii, obj in enumerate(inventory.all_components.collection):
        for coord in ("x", "y", "z"):
            nc.iloc[ii::n_all, nc.columns.get_loc(coord)] = obj.coordinate[coord]
        nodes = np.linspace(ii, ii + conductor.total_elements, n_cs, dtype=int)
        cm.iloc[ii::n_all, 0] = nodes[:-1]
        cm.iloc[ii::n_all, 1] = nodes[1:]

    node_distance = (
        ((nc.iloc[cm["end"], :] - nc.iloc[cm["start"], :]) ** 2).sum(axis=1).apply(np.sqrt)
    )
    if inventory.strands.number == 1:
        node_distance = node_distance / inventory.strands.collection[0].inputs.cos_theta

    lmod = (
        (
            (
                nc.iloc[cm.loc["StrandComponent", "end"], :]
                - nc.iloc[cm.loc["StrandComponent", "start"], :]
            )
            ** 2
        )
        .sum(axis=1)
        .apply(np.sqrt)
    )
    return nc, cm, node_distance, lmod


def legacy_vertex_distance(nc, cm, key, ii, jj):
    cols = key.split("_")
    cm_s = cm.loc["StrandComponent"]
    return (
        (
            (
                nc.iloc[cm_s.iloc[jj, cm_s.columns.get_loc(cols[0])], :]
                - nc.iloc[cm_s.iat[ii, cm_s.columns.get_loc(cols[1])], :]
            )
            ** 2
        )
        .sum(axis="columns")
        .apply(np.sqrt)
        .to_numpy()
    )


def legacy_transversal_distance(nc, contact_nodes):
    contact = pd.DataFrame(contact_nodes, columns=["start", "end"])
    strand_coords = nc.loc["StrandComponent"]
    return (
        (
            (
                strand_coords.iloc[contact["end"], :].reset_index(level="Identifier", drop=True)
                - strand_coords.iloc[contact["start"], :].reset_index(level="Identifier", drop=True)
            )
            ** 2
        )
        .sum(axis="columns")
        .apply(np.sqrt)
        .to_numpy()
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_strands", [1, 2, 3])
def test_tables_match_the_legacy_pandas_tables_bitwise(n_strands):
    conductor = build_tables(make_conductor(n_strands=n_strands))
    nc, cm, node_distance, lmod = legacy_tables(conductor)

    assert conductor.nodal_coordinates.shape == (conductor.total_nodes, 3)
    np.testing.assert_array_equal(conductor.nodal_coordinates, nc.to_numpy())
    np.testing.assert_array_equal(conductor.connectivity_matrix, cm.to_numpy())
    assert conductor.connectivity_matrix.dtype == np.int64
    np.testing.assert_array_equal(conductor.node_distance, node_distance.to_numpy())

    strand_mask = np.asarray(nc.index.get_level_values("Kind") == "StrandComponent")
    np.testing.assert_array_equal(conductor.strand_node_mask, strand_mask)
    np.testing.assert_array_equal(
        conductor.strand_element_mask, strand_mask[: -conductor.inventory.all_components.number]
    )

    connectivity = inductance._strand_connectivity(conductor)
    np.testing.assert_array_equal(
        inductance.row_norms(inductance._element_vectors(conductor, connectivity)),
        lmod.to_numpy(),
    )


def test_component_element_lengths_is_the_legacy_label_lookup():
    conductor = build_tables(make_conductor(n_strands=2))
    nc, cm, node_distance, lmod = legacy_tables(conductor)
    for obj in conductor.inventory.all_components.collection:
        kind = "StrandComponent" if isinstance(obj, StrandComponent) else None
        lengths = Conductor.component_element_lengths(conductor, obj.identifier)
        assert lengths.shape == (conductor.mesh.number_of_elements,)
        if kind:
            np.testing.assert_array_equal(lengths, node_distance[(kind, obj.identifier)].to_numpy())


def test_vertex_distances_match_the_legacy_implementation_bitwise():
    conductor = build_tables(make_conductor(n_strands=3, n_elements=5))
    nc, cm, _, _ = legacy_tables(conductor)
    connectivity = inductance._strand_connectivity(conductor)
    n_cc = conductor.electric.total_elements_current_carriers
    for ii in range(n_cc - 1):
        jj = np.r_[ii + 1 : n_cc]
        for key in ("end_end", "end_start", "start_start", "start_end"):
            np.testing.assert_array_equal(
                inductance._vertex_to_vertex_distance(conductor, connectivity, key, ii, jj),
                legacy_vertex_distance(nc, cm, key, ii, jj),
            )


def test_contact_nodes_and_transversal_distance():
    conductor = build_tables(make_conductor(n_strands=3, n_elements=4))
    contact_nodes = conductor.electric.contact_nodes_current_carriers
    n_pairs_first = 3 * 2
    assert contact_nodes.shape == (n_pairs_first * 5, 2)
    assert contact_nodes.dtype == np.int64
    # cross-section major: the second cross section is offset by n_strands
    np.testing.assert_array_equal(
        contact_nodes[n_pairs_first : 2 * n_pairs_first], contact_nodes[:n_pairs_first] + 3
    )
    nc, *_ = legacy_tables(conductor)
    np.testing.assert_array_equal(
        conductance.evaluate_transversal_distance(conductor),
        legacy_transversal_distance(nc, contact_nodes),
    )
    # the two strands of a pair sit on the same helix radius: chord length
    expected = 2.0 * 0.01 * np.sin(np.pi / 3)
    np.testing.assert_allclose(conductance.evaluate_transversal_distance(conductor), expected)


def test_no_contacts_gives_an_empty_two_column_array():
    conductor = make_conductor(n_strands=2)
    conductor.coupling.contact_perimeter_flag.matrix[:] = 0.0
    build_tables(conductor)
    assert conductor.electric.contact_nodes_current_carriers.shape == (0, 2)
    assert conductor.electric.contact_nodes_current_carriers.dtype == np.int64
    assert conductance.evaluate_transversal_distance(conductor).shape == (0,)


def test_analytical_inductance_runs_and_agrees_with_the_quadrature_path():
    conductor = build_tables(make_conductor(n_strands=2, n_elements=4))
    with np.errstate(divide="ignore", invalid="ignore"):
        inductance.build_inductance_matrix(conductor)
    analytical = conductor.electric.inductance_matrix.copy()
    assert np.all(np.isfinite(analytical))
    np.testing.assert_array_equal(analytical, analytical.T)
    assert np.all(np.diag(analytical) > 0.0)

    conductor.operations.inductance_mode = InductanceMode.APPROXIMATED
    inductance.build_inductance_matrix(conductor)
    approximate = conductor.electric.inductance_matrix

    # identical self-inductance formula on both paths
    np.testing.assert_array_equal(np.diag(analytical), np.diag(approximate))
    # mutual terms of elements that do not share a node: Neumann closed form
    # versus numerical double integration
    connectivity = inductance._strand_connectivity(conductor)
    n_cc = connectivity.shape[0]
    for ii in range(n_cc):
        for jj in range(ii + 1, n_cc):
            if set(connectivity[ii]) & set(connectivity[jj]):
                continue
            assert analytical[ii, jj] == pytest.approx(approximate[ii, jj], rel=2e-3, abs=1e-12)
