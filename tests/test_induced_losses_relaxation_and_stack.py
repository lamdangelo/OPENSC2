"""Tests of (i) the relaxation coupling-loss model with loop families and
(ii) the thin-strip hysteresis loss of REBCO stacks with the stack shielding
bound (both added after the review of docs/hts_induced_losses.tex)."""

from types import SimpleNamespace

import numpy as np
import pytest

from components.solid.strand_component import StrandComponent
from components.solid.stack_component import StackComponent
from physical_fields.physical_field import FieldContainer, GridLocation

MU0 = 4.0e-7 * np.pi
NODES = 7
AREA = 3.0e-5  # m^2 (one stack)


def make_strand(time_constant, relaxation_time=0.0, **extra):
    strand = StrandComponent.__new__(StrandComponent)
    strand.identifier = "S1"
    strand.node_fields = FieldContainer(GridLocation.NODE)
    strand.operations = SimpleNamespace(
        coupling_loss_time_constant=time_constant,
        coupling_loss_relaxation_time=relaxation_time,
        coupling_loss_copper_scaling=False,
        coupling_loss_reference_temperature=0.0,
        coupling_loss_reference_field=0.0,
        **extra,
    )
    strand.inputs = SimpleNamespace(cross_section=AREA, residual_resistivity_ratio=100.0)
    return strand


def conductor_at(time):
    return SimpleNamespace(cond_time=[time], mesh=SimpleNamespace(number_of_nodes=NODES))


def drive(strand, times, fields):
    """Feed a field history B(t_k); return the power after each call."""
    powers = []
    for time, field in zip(times, fields):
        strand.node_fields.B_field = np.full(NODES, field)
        strand.node_fields.temperature = np.full(NODES, 4.5)
        strand.get_coupling_loss(conductor_at(time))
        powers.append(float(strand.node_fields.coupling_loss_linear_power[0, 0]))
    return np.array(powers)


def lumped_reference(n_tau, rate):
    return n_tau / MU0 * rate**2 * AREA


# --------------------------------------------------------------------------
# Relaxation model
# --------------------------------------------------------------------------

def test_zero_relaxation_time_is_the_lumped_model_bit_for_bit():
    times = [0.0, 0.01, 0.02]
    fields = [10.0, 9.8, 9.5]
    new = drive(make_strand(0.3, 0.0), times, fields)
    # the pre-existing expression, evaluated in the same order
    rates = [0.0, (9.8 - 10.0) / 0.01, (9.5 - 9.8) / 0.01]
    old = np.array([0.3 / MU0 * r**2 * AREA for r in rates])
    np.testing.assert_array_equal(new, old)


def test_relaxation_energy_fraction_matches_closed_form():
    """Exponential dump B = B0 exp(-t/tau_d): with M solving
    tau dM/dt + M = -(n tau/mu0) Bdot the deposited energy per unit strand
    volume is n*tau/(tau + tau_d) * B0^2/(2 mu0) (closed-form integral of
    mu0 M^2/(n tau)), versus n*tau/tau_d * B0^2/(2 mu0) for the lumped
    model: the loops shield when they are slower than the field change."""
    n_tau, tau, tau_d, b0 = 0.4, 0.25, 0.65, 16.0
    dt = 2.0e-4
    times = np.arange(0.0, 12.0 * tau_d, dt)
    fields = b0 * np.exp(-times / tau_d)
    # the rate at sample k is (B_k - B_{k-1})/dt: energy = sum p_k * dt
    strand = make_strand(n_tau, tau)
    powers = drive(strand, times, fields)
    energy = float(np.sum(powers) * dt) / AREA
    expected = n_tau / (tau + tau_d) * b0**2 / (2.0 * MU0)
    assert energy == pytest.approx(expected, rel=2.0e-3)
    lumped = drive(make_strand(n_tau, 0.0), times, fields)
    assert float(np.sum(lumped) * dt) / AREA == pytest.approx(n_tau / tau_d * b0**2 / (2.0 * MU0), rel=2.0e-3)


def test_relaxation_magnetisation_follows_first_order_response():
    """Constant rate switched on: M(t) = M_inf (1 - exp(-t/tau)) with
    M_inf = -(n tau/mu0) Bdot; the power then rises as (1 - e^-t/tau)^2."""
    n_tau, tau, rate, dt = 0.3, 0.1, -20.0, 1.0e-3
    times = np.arange(0.0, 0.6, dt)
    fields = 10.0 + rate * times
    powers = drive(make_strand(n_tau, tau), times, fields)
    steady = lumped_reference(n_tau, rate)
    t = times[1:]  # the first sample has no rate
    expected = steady * (1.0 - np.exp(-t / tau)) ** 2
    # backward Euler lags the exact response by ~dt/2: compare loosely
    np.testing.assert_allclose(powers[1:], expected, rtol=0.05, atol=1e-3 * steady)
    assert powers[-1] == pytest.approx(steady, rel=1e-2)


def test_two_families_sum_and_scalar_relaxation_broadcasts():
    times = [0.0, 0.01, 0.02, 0.03]
    fields = [10.0, 9.8, 9.6, 9.4]
    both = drive(make_strand([0.3, 0.05], 0.0), times, fields)
    first = drive(make_strand(0.3, 0.0), times, fields)
    second = drive(make_strand(0.05, 0.0), times, fields)
    np.testing.assert_allclose(both, first + second, rtol=1e-12)
    mixed = make_strand([0.3, 0.05], [0.2, 0.0])
    assert mixed._coupling_loss_families() == [(0.3, 0.2), (0.05, 0.0)]
    with pytest.raises(ValueError):
        make_strand([0.3, 0.05], [0.2, 0.1, 0.0])._coupling_loss_families()


def test_disabled_family_is_dropped():
    assert make_strand(0.0, 0.5)._coupling_loss_families() == []
    assert make_strand([0.0, 0.2], [0.5, 0.1])._coupling_loss_families() == [(0.2, 0.1)]


def test_copper_scaling_follows_nist_resistivity():
    from properties_of_materials.electrical_conductivity import electrical_resistivity_of

    strand = make_strand(0.3, 0.0)
    strand.operations.coupling_loss_copper_scaling = True
    strand.operations.coupling_loss_reference_temperature = 4.5
    strand.operations.coupling_loss_reference_field = 8.4
    times, fields = [0.0, 0.01], [8.4, 8.2]
    scaled = drive(strand, times, fields)
    reference = drive(make_strand(0.3, 0.0), times, fields)
    rho_ref = electrical_resistivity_of("cu", np.array([4.5]), magnetic_field=np.array([8.4]), residual_resistivity_ratio=100.0)[0]
    rho = electrical_resistivity_of("cu", np.array([4.5]), magnetic_field=np.array([8.2]), residual_resistivity_ratio=100.0)[0]
    assert scaled[1] == pytest.approx(reference[1] * rho_ref / rho, rel=1e-12)
    assert rho < rho_ref  # lower field, lower magnetoresistance


# --------------------------------------------------------------------------
# Stack hysteresis
# --------------------------------------------------------------------------

def make_stack(tape_hysteresis=True, filament_diameter=0.0):
    stack = StackComponent.__new__(StackComponent)
    stack.identifier = "STACK1"
    stack.node_fields = FieldContainer(GridLocation.NODE)
    stack.operations = SimpleNamespace(
        tape_hysteresis_loss=tape_hysteresis, filament_diameter=filament_diameter
    )
    n_tapes, width, t_hts = 80, 6.0e-3, 2.0e-6
    stack.inputs = SimpleNamespace(
        superconducting_material="ybco",
        cross_section=AREA,
        stack_width=width,
        tape_identifier=str(n_tapes),
        critical_current_scaling_constant=1.838522e8,
        critical_temperature_at_0T=93.0,
        upper_critical_field_at_0K=140.0,
    )
    stack.sc = n_tapes * width * t_hts
    stack.sc_cross_section = stack.sc
    return stack


def drive_stack(stack, b0, b1, dt, temperature=4.5):
    stack.node_fields.temperature = np.full(NODES, temperature)
    stack.node_fields.B_field = np.full(NODES, b0)
    stack.get_hysteresis_loss(conductor_at(0.0))
    stack.node_fields.B_field = np.full(NODES, b1)
    stack.get_hysteresis_loss(conductor_at(dt))
    return float(stack.node_fields.hysteresis_loss_linear_power[0, 0])


def test_stack_hysteresis_matches_strip_formula_at_high_field():
    from properties_of_materials.rare_earth_123 import critical_current_density_re123

    stack = make_stack()
    b0, rate, dt = 14.1, -24.9, 1.0e-3
    power = drive_stack(stack, b0, b0 + rate * dt, dt)
    jc = critical_current_density_re123(np.array([4.5]), np.array([b0 + rate * dt]), 93.0, 140.0, 1.838522e8)[0]
    i_c = jc * 6.0e-3 * 2.0e-6
    magnetisation = (2.0 / np.pi) * 80 * i_c * 6.0e-3 / 4.0
    assert i_c == pytest.approx(500.0, rel=0.05)  # the deck calibration
    assert magnetisation < (b0 + rate * dt) * AREA / MU0  # strip value binds, not the bound
    assert power == pytest.approx(magnetisation * abs(rate), rel=1e-12)
    assert 800.0 < power < 1100.0  # ~950 W/m per stack at the DP1 peak rate


def test_stack_hysteresis_is_bounded_by_full_shielding_at_low_field():
    stack = make_stack()
    b0, rate, dt = 1.0, -24.9, 1.0e-3
    power = drive_stack(stack, b0, b0 + rate * dt, dt)
    bound = (b0 + rate * dt) * AREA / MU0 * abs(rate)
    assert power == pytest.approx(bound, rel=1e-12)
    # finite and vanishing as B -> 0 (the Jc fit alone would diverge)
    tiny = drive_stack(make_stack(), 1.0e-3, 1.0e-3 + rate * dt, dt)
    assert np.isfinite(tiny) and tiny < power


def test_stack_hysteresis_key_off_falls_back_to_base_model():
    stack = make_stack(tape_hysteresis=False, filament_diameter=0.0)
    power = drive_stack(stack, 14.0, 13.9, 1.0e-3)
    assert power == 0.0  # base model with d_f = 0 is disabled
