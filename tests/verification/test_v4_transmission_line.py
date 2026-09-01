"""V4 — Coupled channel + lumped termination: transmission-line reflection.

The production coupled solve (YAML hydraulic_network section ->
build_coupled_network -> per-step bordered Schur solve in
solve_coupled_conductors_step) terminates the channel outlet port on a
single linear network resistance R_term to a reference reservoir. A compact
raised-cosine pressure pulse launched from the inlet reflects off the
termination with the transmission-line reflection coefficient

    Gamma = (R_term - Z) / (R_term + Z),   Z = a/A  (see analytical.py)

(standard transmission-line theory). These four runs are the direct
quantitative test of the port coupling: the incident pulse only reflects
with the analytical Gamma if the Schur-complement condensation preserves
pressure continuity (p_port = p_node) and mass conservation
(port flow = termination branch flow) at the interface — any leak or
pressure defect at the port shows up as a spurious reflection, which the
matched run (Gamma = 0) bounds below 2% of the incident amplitude.

Gamma is measured as the ratio of time-integrated pulse areas over disjoint
incident/reflected windows at the 3L/4 probe, with the windows computed
from the wave speed and the geometry (not hardcoded). The area of the pulse
is invariant under the scheme's diffusive pulse spreading, while the peak
amplitude is not (the measured peak ratio for a perfectly reflecting end is
0.89 purely from spreading), so the area ratio isolates the physical
reflection; the residual estimator bias measured on the perfectly known
closed/open cases is ~0.3%, well inside the 2% tolerances.

The inlet keeps its imposed-pressure boundary row (IMPOSE_PRESSURE_DROP);
the outlet pressure row is replaced by the port continuity constraint. The
inlet is biased +1e-4 Pa above the outlet so the consistent initialization
does not evaluate the (unguarded) Blasius correlation at exactly zero flow
— see the note in test_v3_water_hammer.py; the bias is 1e-7 of the pulse.
"""

import numpy as np
import pytest

import analytical
from case_builders import (
    ACOUSTIC_FLUID,
    CHANNEL_LENGTH,
    CHARACTERISTIC_IMPEDANCE,
    WAVE_SPEED,
    ChannelProbe,
    StepDriver,
    channel_operations,
    run_case,
    write_channel_run_directory,
)

BASE_PRESSURE = 1.0e5  # Pa
INLET_BIAS = 1.0e-4  # Pa (avoids the exact-zero-flow initialization NaN)
PULSE_START = 0.05  # s
PULSE_WIDTH = 0.0633  # s = 2 m at a = 31.6 m/s: 80 elements on the 401 mesh
PULSE_AMPLITUDE = 1.0e3  # Pa
TIME_STEP = 1.0e-3  # s (same mesh/dt pairing as V3a: CFL = 1.27)
NUMBER_OF_ELEMENTS = 401
PROBE_INDEX = 301  # node nearest 3L/4 on the 402-node mesh
# RELIEF_VALVE_BLOCKED_RESISTANCE-scale closed termination.
BLOCKED_RESISTANCE = 1.0e12  # Pa/(kg/s)


def termination_network(termination_resistance):
    """Zero-volume internal junction (purely resistive termination) ->
    linear valve R_term -> reference reservoir; channel outlet ported to the
    junction."""
    return {
        "fluid_type": "constant",
        "nodes": [
            {
                "identifier": "terminal",
                "kind": "internal",
                "initial_pressure": BASE_PRESSURE,
                "initial_temperature": 300.0,
            },
            {
                "identifier": "sink",
                "kind": "reservoir",
                "pressure": BASE_PRESSURE,
                "temperature": 300.0,
            },
        ],
        "branches": [
            {
                "identifier": "termination",
                "kind": "valve",
                "from": "terminal",
                "to": "sink",
                "linear_resistance": float(termination_resistance),
            },
        ],
        "ports": [
            {
                "node": "terminal",
                "conductor": "CONDUCTOR_1",
                "channel": "CHAN_1",
                "end": "outlet",
            },
        ],
    }


def run_termination(tmp_path, name, termination_resistance):
    """Coupled run; returns (t, p_probe, baseline, incident window mask,
    reflected window mask)."""
    run_directory = write_channel_run_directory(
        tmp_path / name,
        number_of_elements=NUMBER_OF_ELEMENTS,
        time_step=TIME_STEP,
        end_time=0.75,
        method="BDF2",
        hydraulic_boundary_condition=1,
        inlet_pressure=BASE_PRESSURE + INLET_BIAS,
        outlet_pressure=BASE_PRESSURE,
        initial_pressure=BASE_PRESSURE,
        inlet_mass_rate=0.0,
        outlet_mass_rate=0.0,
        network_section=termination_network(termination_resistance),
    )

    def inlet_pulse(simulation):
        t_next = simulation.simulation_time[-1] + TIME_STEP
        channel_operations(simulation).inlet_pressure = (
            BASE_PRESSURE
            + INLET_BIAS
            + float(
                analytical.raised_cosine_pulse(
                    np.asarray(t_next), PULSE_START, PULSE_WIDTH, PULSE_AMPLITUDE
                )
            )
        )

    probe = ChannelProbe(node_index=PROBE_INDEX)
    simulation = run_case(
        run_directory, step_callback=StepDriver(inlet_pulse, probe)
    )
    probe.record_final(simulation)
    times, pressures, _ = probe.as_arrays()

    baseline = pressures[(times > 0.005) & (times < PULSE_START)].mean()

    # Windows from the wave speed and the geometry: the pulse center leaves
    # the inlet at PULSE_START + w/2, passes the 3L/4 probe after 0.75 L/a
    # (incident) and again after (L + L/4)/a (reflected). Half-width 0.08 s
    # covers the pulse (0.063 s) plus its diffusive spreading; the windows
    # are disjoint, and the inlet re-reflection reaches the probe only at
    # +2.25 L/a = 0.95 s > end of the reflected window.
    pulse_center = PULSE_START + 0.5 * PULSE_WIDTH
    incident_center = pulse_center + 0.75 * CHANNEL_LENGTH / WAVE_SPEED
    reflected_center = pulse_center + 1.25 * CHANNEL_LENGTH / WAVE_SPEED
    incident = np.abs(times - incident_center) < 0.08
    reflected = np.abs(times - reflected_center) < 0.08
    return times, pressures, baseline, incident, reflected


def measured_gamma(times, pressures, baseline, incident, reflected):
    """Reflection coefficient as the signed ratio of time-integrated pulse
    areas (invariant under diffusive spreading, unlike the peak ratio)."""
    incident_area = np.trapezoid(pressures[incident] - baseline, times[incident])
    reflected_area = np.trapezoid(pressures[reflected] - baseline, times[reflected])
    return reflected_area / incident_area


def test_matched_termination_absorbs_pulse(acoustic_fluid, tmp_path):
    # R_term = Z: the termination is impedance-matched and the pulse must be
    # absorbed. Any residual signal in the reflected window is a direct
    # measure of a port pressure/mass defect or of the discrete impedance
    # mismatch; 2% of the incident peak (spec) is ~7x the measured 0.27%.
    times, pressures, baseline, incident, reflected = run_termination(
        tmp_path, "matched", CHARACTERISTIC_IMPEDANCE
    )
    incident_peak = np.max(np.abs(pressures[incident] - baseline))
    reflected_peak = np.max(np.abs(pressures[reflected] - baseline))
    assert reflected_peak < 0.02 * incident_peak


@pytest.mark.parametrize(
    "name, termination_resistance, expected_gamma",
    [
        # Closed end: R_term at the relief-valve blocked-resistance scale.
        ("closed", BLOCKED_RESISTANCE, 1.0),
        # Reservoir (open) end: tiny but finite linear resistance.
        ("open", 1.0e-4 * CHARACTERISTIC_IMPEDANCE, -1.0),
        # Intermediate: Gamma = (3Z - Z)/(3Z + Z) = 0.5.
        ("three_z", 3.0 * CHARACTERISTIC_IMPEDANCE, 0.5),
    ],
)
def test_reflection_coefficient(
    acoustic_fluid, tmp_path, name, termination_resistance, expected_gamma
):
    # 0.02 absolute on Gamma (spec): the area estimator's bias measured on
    # the exactly-known closed/open limits is ~0.3% (window truncation of
    # the diffused tails), and the analytical Gamma for the finite "open"
    # resistance differs from -1 by only 2e-4.
    times, pressures, baseline, incident, reflected = run_termination(
        tmp_path, name, termination_resistance
    )
    gamma = measured_gamma(times, pressures, baseline, incident, reflected)
    analytical_gamma = analytical.reflection_coefficient(
        termination_resistance, CHARACTERISTIC_IMPEDANCE
    )
    assert analytical_gamma == pytest.approx(expected_gamma, abs=2.0e-4)
    assert gamma == pytest.approx(analytical_gamma, abs=0.02)
