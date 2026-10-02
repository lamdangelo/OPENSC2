"""Unit tests of the AC coupling-loss heat source on strand components.

The lumped effective-time-constant model dissipates

    p = (n·tau_eff / mu0) * (dB/dt)^2        [W/m^3 of composite strand]

so the linear power is p * cross_section. dB/dt is evaluated numerically
from the nodal field stored at the previous thermal step. The tests drive
a bare StrandComponent with a prescribed field history and check the
power against the closed form, the energy of a linear ramp, and the
opt-in behaviour (zero time constant produces exactly zero power).
"""

from types import SimpleNamespace

import numpy as np

from components.solid.strand_component import StrandComponent
from physical_fields.physical_field import FieldContainer, GridLocation

NUMBER_OF_NODES = 11
CROSS_SECTION = 6.2e-5  # m^2
TIME_CONSTANT = 0.05  # n·tau_eff in s


def make_strand(time_constant: float) -> StrandComponent:
    """Bare StrandComponent with only what get_coupling_loss touches."""
    strand = StrandComponent.__new__(StrandComponent)
    strand.node_fields = FieldContainer(GridLocation.NODE)
    strand.operations = SimpleNamespace(
        coupling_loss_time_constant=time_constant
    )
    strand.inputs = SimpleNamespace(cross_section=CROSS_SECTION)
    return strand


def make_conductor(time: float) -> SimpleNamespace:
    return SimpleNamespace(
        cond_time=[time],
        mesh=SimpleNamespace(number_of_nodes=NUMBER_OF_NODES),
    )


def test_coupling_loss_matches_closed_form():
    """A uniform field ramp must give p = (n·tau/mu0)*(dB/dt)^2*A exactly."""
    strand = make_strand(TIME_CONSTANT)
    field_rate = -2.5  # T/s, dump-like decay
    time_step = 0.01

    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 6.0)
    strand.get_coupling_loss(make_conductor(0.0))
    # First call only allocates and stores the field: power must be zero.
    assert np.all(strand.node_fields.coupling_loss_linear_power == 0.0)

    strand.node_fields.B_field = 6.0 + field_rate * time_step * np.ones(
        NUMBER_OF_NODES
    )
    strand.get_coupling_loss(make_conductor(time_step))

    expected = (
        TIME_CONSTANT / StrandComponent.MU0 * field_rate**2 * CROSS_SECTION
    )
    power = strand.node_fields.coupling_loss_linear_power
    assert power.shape == (NUMBER_OF_NODES, 1)
    assert np.allclose(power, expected, rtol=1e-12)


def test_coupling_loss_energy_of_linear_ramp():
    """Stepping through a linear ramp must integrate to the exact energy.

    For B going from B0 to 0 linearly over T, the exact deposited energy
    per unit length is (n·tau/mu0)*(B0/T)^2*A*T, independent of the step
    subdivision because dB/dt is constant.
    """
    strand = make_strand(TIME_CONSTANT)
    ramp_duration = 2.0
    initial_field = 6.0
    times = np.linspace(0.0, ramp_duration, 41)

    energy = 0.0
    previous_time = times[0]
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, initial_field)
    strand.get_coupling_loss(make_conductor(times[0]))
    for time in times[1:]:
        strand.node_fields.B_field = np.full(
            NUMBER_OF_NODES, initial_field * (1.0 - time / ramp_duration)
        )
        strand.get_coupling_loss(make_conductor(time))
        energy += strand.node_fields.coupling_loss_linear_power[0, 0] * (
            time - previous_time
        )
        previous_time = time

    exact = (
        TIME_CONSTANT
        / StrandComponent.MU0
        * (initial_field / ramp_duration) ** 2
        * CROSS_SECTION
        * ramp_duration
    )
    assert np.isclose(energy, exact, rtol=1e-10)


def test_coupling_loss_disabled_by_default():
    """Zero time constant must keep the power identically zero."""
    strand = make_strand(0.0)
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 6.0)
    strand.get_coupling_loss(make_conductor(0.0))
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 3.0)
    strand.get_coupling_loss(make_conductor(1.0))
    assert np.all(strand.node_fields.coupling_loss_linear_power == 0.0)


def test_coupling_loss_holds_value_on_repeated_time():
    """A second call at the same time must not divide by zero nor update."""
    strand = make_strand(TIME_CONSTANT)
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 6.0)
    strand.get_coupling_loss(make_conductor(0.0))
    strand.node_fields.B_field = np.full(NUMBER_OF_NODES, 5.0)
    strand.get_coupling_loss(make_conductor(0.5))
    power_first = strand.node_fields.coupling_loss_linear_power.copy()
    strand.get_coupling_loss(make_conductor(0.5))
    assert np.array_equal(
        strand.node_fields.coupling_loss_linear_power, power_first
    )
