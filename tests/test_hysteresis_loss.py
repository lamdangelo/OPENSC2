"""Unit tests of the superconductor hysteresis (persistent-current) loss.

Fully-penetrated critical-state model, per unit superconductor volume,

    p = (2 / 3pi) * Jc(B, T) * d_f * |dB/dt|

so the linear power is p * A_sc, A_sc = cross_section / (1 + Cu:SC ratio).
Unlike the coupling and eddy losses (~ (dB/dt)^2), this is linear in
|dB/dt|. dB/dt is evaluated from the field stored at the previous thermal
step. The tests drive a bare StrandComponent with a prescribed field and
check the power against the closed form, the |dB/dt| (not squared) scaling,
and the opt-in behaviour.
"""

from types import SimpleNamespace

import numpy as np

from components.solid.strand_component import StrandComponent
from physical_fields.physical_field import FieldContainer, GridLocation
from properties_of_materials.niobium_titanium_w7x import (
    critical_current_density_nbti_w7x,
)

NUMBER_OF_NODES = 9
CROSS_SECTION = 6.2004e-5  # m^2
CU_SC_RATIO = 2.6
FILAMENT_DIAMETER = 26e-6  # m, W7-X NbTi
BC20, C0, TC0 = 14.61, 16.8512e10, 9.03  # NbTi-W7X critical surface


def make_strand(filament_diameter: float) -> StrandComponent:
    strand = StrandComponent.__new__(StrandComponent)
    strand.node_fields = FieldContainer(GridLocation.NODE)
    strand.operations = SimpleNamespace(filament_diameter=filament_diameter)
    strand.inputs = SimpleNamespace(
        superconducting_material="nbti-w7x",
        cross_section=CROSS_SECTION,
        stabilizer_to_sc_ratio=CU_SC_RATIO,
        upper_critical_field_at_0K=BC20,
        critical_current_scaling_constant=C0,
        critical_temperature_at_0T=TC0,
    )
    return strand


def conductor_at(time: float) -> SimpleNamespace:
    return SimpleNamespace(
        cond_time=[time],
        mesh=SimpleNamespace(number_of_nodes=NUMBER_OF_NODES),
    )


def drive(strand: StrandComponent, b0: float, b1: float, dt: float):
    strand.node_fields.temperature = np.full(NUMBER_OF_NODES, 6.0)
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, b0)
    strand.get_hysteresis_loss(conductor_at(0.0))
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, b1)
    strand.get_hysteresis_loss(conductor_at(dt))


def test_hysteresis_matches_closed_form():
    strand = make_strand(FILAMENT_DIAMETER)
    b0, dt, field_rate = 4.35, 0.02, -3.0
    b1 = b0 + field_rate * dt
    drive(strand, b0, b1, dt)

    temperature = np.full(NUMBER_OF_NODES, 6.0)
    field = np.full(NUMBER_OF_NODES, b1)
    jc = critical_current_density_nbti_w7x(temperature, field, BC20, C0, TC0)
    a_sc = CROSS_SECTION / (1.0 + CU_SC_RATIO)
    expected = (
        (2.0 / (3.0 * np.pi))
        * jc
        * FILAMENT_DIAMETER
        * abs(field_rate)
        * a_sc
    )
    power = strand.node_fields.hysteresis_loss_linear_power
    assert power.shape == (NUMBER_OF_NODES, 1)
    assert np.allclose(power[:, 0], expected, rtol=1e-12)


def test_hysteresis_is_linear_in_rate():
    """Halving dt doubles |dB/dt| and doubles the power (linear, not the
    (dB/dt)^2 of the coupling/eddy losses). Final field fixed so Jc(B) is
    the same in both cases."""

    def power_for(dt: float) -> float:
        strand = make_strand(FILAMENT_DIAMETER)
        drive(strand, 4.02, 4.0, dt)  # fixed final B=4.0, |rate|=0.02/dt
        return float(strand.node_fields.hysteresis_loss_linear_power[0, 0])

    assert abs(power_for(0.01) - 2.0 * power_for(0.02)) < 1e-9 * power_for(0.01)


def test_hysteresis_depends_on_abs_rate():
    """Same |dB/dt| and same final field, opposite sign -> identical power."""

    def power_for(b0: float) -> float:
        strand = make_strand(FILAMENT_DIAMETER)
        drive(strand, b0, 4.0, 0.01)  # final B=4.0; rate = (4.0 - b0)/0.01
        return float(strand.node_fields.hysteresis_loss_linear_power[0, 0])

    # b0=3.98 -> rate +2 ; b0=4.02 -> rate -2 ; |rate|=2 and final B=4.0 both
    assert abs(power_for(3.98) - power_for(4.02)) < 1e-9 * power_for(3.98)


def test_hysteresis_disabled_is_zero():
    strand = make_strand(0.0)
    drive(strand, 4.35, 3.0, 0.01)
    assert np.all(strand.node_fields.hysteresis_loss_linear_power == 0.0)


def test_hysteresis_unsupported_material_raises():
    import pytest

    strand = make_strand(FILAMENT_DIAMETER)
    strand.inputs.superconducting_material = "nb3sn"
    with pytest.raises(NotImplementedError):
        drive(strand, 4.35, 3.0, 0.01)
