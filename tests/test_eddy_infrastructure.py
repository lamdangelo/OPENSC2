"""Step-1 infrastructure for the eddy-current heat sources.

Covers the shared nodal dB/dt helper ``SolidComponent._field_rate`` (used by
the AC coupling loss and, later, the jacket/copper eddy losses) and the
material-name conductivity dispatcher. No physics is wired to a solver here;
these are the reusable pieces the eddy terms will build on.
"""

from types import SimpleNamespace

import numpy as np

from components.solid.strand_component import StrandComponent
from components.jacket.jacket_component import JacketComponent
from physical_fields.physical_field import FieldContainer, GridLocation
from properties_of_materials.electrical_conductivity import (
    electrical_conductivity_of,
    electrical_resistivity_of,
)

NUMBER_OF_NODES = 7
JACKET_GEOMETRY_CONSTANT = 4.4437e-9  # m^4, W7-X Al6063 conduit second moment


def make_component() -> StrandComponent:
    component = StrandComponent.__new__(StrandComponent)
    component.node_fields = FieldContainer(GridLocation.NODE)
    return component


def conductor_at(time: float) -> SimpleNamespace:
    return SimpleNamespace(cond_time=[time])


# --------------------------------------------------------------------------
# _field_rate
# --------------------------------------------------------------------------
def test_field_rate_first_call_is_zero_and_seeds_history():
    component = make_component()
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0)
    rate = component._field_rate(conductor_at(0.0))
    assert np.all(rate == 0.0)
    assert rate.shape == (NUMBER_OF_NODES,)


def test_field_rate_matches_finite_difference():
    component = make_component()
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0)
    component._field_rate(conductor_at(0.0))  # seed
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0 - 2.5 * 0.01)
    rate = component._field_rate(conductor_at(0.01))
    assert np.allclose(rate, -2.5, rtol=1e-12)


def test_field_rate_cached_within_a_step():
    """Repeated calls at the same time return one consistent rate and do
    not re-advance the stored field (so several loss terms can share it)."""
    component = make_component()
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0)
    component._field_rate(conductor_at(0.0))
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 3.0)
    first = component._field_rate(conductor_at(0.01)).copy()
    # A second query at the same step, even after B_field moves again,
    # must return the already-computed rate, not a new finite difference.
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 99.0)
    second = component._field_rate(conductor_at(0.01))
    assert np.allclose(first, -100.0, rtol=1e-12)
    assert np.array_equal(first, second)


def test_field_rate_zero_for_nonpositive_step():
    component = make_component()
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0)
    component._field_rate(conductor_at(1.0))
    component.node_fields.B_field = np.full(NUMBER_OF_NODES, 5.0)
    rate = component._field_rate(conductor_at(1.0 - 0.5))  # time went backward
    assert np.all(rate == 0.0)


# --------------------------------------------------------------------------
# electrical conductivity dispatcher
# --------------------------------------------------------------------------
def test_conductivity_is_reciprocal_of_resistivity():
    temperature = np.array([6.0])
    rho = electrical_resistivity_of("Al6063", temperature)
    sigma = electrical_conductivity_of("al6063", temperature)
    assert np.allclose(sigma, 1.0 / rho, rtol=1e-12)


def test_aluminium_6063_residual_resistivity():
    # At low T the Al6063 correlation is dominated by the 8.20 nOhm m floor.
    rho = electrical_resistivity_of("Al6063", np.array([4.2]))
    assert np.all(rho > 0.0)
    assert abs(float(rho[0]) - 8.2e-9) < 5e-10


def test_copper_requires_field_and_rrr():
    import pytest

    with pytest.raises(ValueError):
        electrical_resistivity_of("Cu", np.array([6.0]))


def test_unknown_material_raises():
    import pytest

    with pytest.raises(KeyError):
        electrical_resistivity_of("unobtainium", np.array([6.0]))


# --------------------------------------------------------------------------
# jacket eddy-current loss
# --------------------------------------------------------------------------
def make_jacket(geometry_constant: float, material: str = "al6063") -> JacketComponent:
    jacket = JacketComponent.__new__(JacketComponent)
    jacket.node_fields = FieldContainer(GridLocation.NODE)
    jacket.operations = SimpleNamespace(
        eddy_loss_geometry_constant=geometry_constant
    )
    jacket.inputs = SimpleNamespace(jacket_material=material)
    return jacket


def conductor_with_mesh(time: float) -> SimpleNamespace:
    return SimpleNamespace(
        cond_time=[time],
        mesh=SimpleNamespace(number_of_nodes=NUMBER_OF_NODES),
    )


def test_jacket_eddy_loss_matches_closed_form():
    """p = sigma_Al6063(T) * (dB/dt)^2 * C, exactly."""
    jacket = make_jacket(JACKET_GEOMETRY_CONSTANT)
    field_rate = -3.0  # T/s
    time_step = 0.02
    temperature = np.full(NUMBER_OF_NODES, 6.0)

    jacket.node_fields.temperature = temperature
    jacket.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.35)
    jacket.get_eddy_loss(conductor_with_mesh(0.0))
    # First call only seeds the field history: power must be zero.
    assert np.all(jacket.node_fields.eddy_loss_linear_power == 0.0)

    jacket.node_fields.B_field = 4.35 + field_rate * time_step * np.ones(
        NUMBER_OF_NODES
    )
    jacket.get_eddy_loss(conductor_with_mesh(time_step))

    sigma = electrical_conductivity_of("al6063", temperature)
    expected = sigma * field_rate**2 * JACKET_GEOMETRY_CONSTANT
    power = jacket.node_fields.eddy_loss_linear_power
    assert power.shape == (NUMBER_OF_NODES, 1)
    assert np.allclose(power[:, 0], expected, rtol=1e-12)


def test_jacket_eddy_loss_scales_quadratically_with_rate():
    """Doubling dB/dt quadruples the power."""

    def power_for(field_rate: float) -> float:
        jacket = make_jacket(JACKET_GEOMETRY_CONSTANT)
        jacket.node_fields.temperature = np.full(NUMBER_OF_NODES, 6.0)
        jacket.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.0)
        jacket.get_eddy_loss(conductor_with_mesh(0.0))
        jacket.node_fields.B_field = np.full(
            NUMBER_OF_NODES, 4.0 + field_rate * 0.01
        )
        jacket.get_eddy_loss(conductor_with_mesh(0.01))
        return float(jacket.node_fields.eddy_loss_linear_power[0, 0])

    assert power_for(2.0) == pytest_approx(4.0 * power_for(1.0))


def test_jacket_eddy_loss_disabled_is_exactly_zero():
    """Zero geometry constant must give exactly zero power (opt-in)."""
    jacket = make_jacket(0.0)
    jacket.node_fields.temperature = np.full(NUMBER_OF_NODES, 6.0)
    jacket.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.35)
    jacket.get_eddy_loss(conductor_with_mesh(0.0))
    jacket.node_fields.B_field = np.full(NUMBER_OF_NODES, 3.0)
    jacket.get_eddy_loss(conductor_with_mesh(0.01))
    assert np.all(jacket.node_fields.eddy_loss_linear_power == 0.0)


def test_base_eddy_conductivity_not_implemented():
    """A solid component without an eddy-conductivity override must fail
    loudly rather than silently produce nonsense."""
    import pytest

    from components.solid.solid_component import SolidComponent

    component = SolidComponent.__new__(SolidComponent)
    with pytest.raises(NotImplementedError):
        component._eddy_conductivity(conductor_with_mesh(0.0))


# --------------------------------------------------------------------------
# strand copper-matrix eddy loss
# --------------------------------------------------------------------------
STRAND_EDDY_GEOMETRY_CONSTANT = 9.0938e-13  # m^4, W7-X 243-strand copper


def make_strand_eddy(
    geometry_constant: float, rrr: float = 160.0
) -> StrandComponent:
    strand = StrandComponent.__new__(StrandComponent)
    strand.node_fields = FieldContainer(GridLocation.NODE)
    strand.operations = SimpleNamespace(
        eddy_loss_geometry_constant=geometry_constant
    )
    strand.inputs = SimpleNamespace(
        stabilizer_material="Cu", residual_resistivity_ratio=rrr
    )
    return strand


def test_strand_copper_eddy_matches_closed_form():
    """p = sigma_Cu(T, B, RRR) * (dB/dt)^2 * C, field- and RRR-dependent."""
    strand = make_strand_eddy(STRAND_EDDY_GEOMETRY_CONSTANT)
    field_rate = -3.0  # T/s
    time_step = 0.02
    temperature = np.full(NUMBER_OF_NODES, 6.0)

    strand.node_fields.temperature = temperature
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.35)
    strand.get_eddy_loss(conductor_with_mesh(0.0))
    assert np.all(strand.node_fields.eddy_loss_linear_power == 0.0)

    field_final = np.full(NUMBER_OF_NODES, 4.35 + field_rate * time_step)
    strand.node_fields.B_field = field_final
    strand.get_eddy_loss(conductor_with_mesh(time_step))

    sigma = electrical_conductivity_of(
        "Cu", temperature, magnetic_field=field_final,
        residual_resistivity_ratio=160.0,
    )
    expected = sigma * field_rate**2 * STRAND_EDDY_GEOMETRY_CONSTANT
    assert np.allclose(
        strand.node_fields.eddy_loss_linear_power[:, 0], expected, rtol=1e-12
    )


def test_strand_copper_eddy_disabled_is_zero():
    strand = make_strand_eddy(0.0)
    strand.node_fields.temperature = np.full(NUMBER_OF_NODES, 6.0)
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 4.35)
    strand.get_eddy_loss(conductor_with_mesh(0.0))
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 3.0)
    strand.get_eddy_loss(conductor_with_mesh(0.01))
    assert np.all(strand.node_fields.eddy_loss_linear_power == 0.0)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-12)
