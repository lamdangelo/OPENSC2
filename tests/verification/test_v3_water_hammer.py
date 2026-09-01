"""V3 — 1D channel alone: water hammer and organ-pipe eigenfrequencies.

The channel model discretizes (v, p, T); linearized about a quiescent state
its pressure/velocity block is the fluid transmission line

    (1/A) dmdot/dt = -dp/dx           inertance per length  l = 1/A
    (A rho kappa_T) dp/dt = -dmdot/dx  capacitance per length c = A rho kappa_T

(with beta = 0 the isentropic and isothermal compressibilities coincide, so
the model's speed_of_sound is exactly a = 1/sqrt(rho0 kappa_T) = sqrt(1/(lc))
per unit... = 31.6228 m/s here). The characteristic impedance in (p, mdot)
variables is Z = sqrt(l/c) = a/A = 4.0264e5 Pa/(kg/s) — the density cancels;
dimensional check: a/A ~ (m/s)/m^2 = 1/(m s) = Pa/(kg/s), and Joukowsky
dp = rho a dv with mdot = rho A v gives the same dp/dmdot = a/A.

All runs go through the production entry point Simulation(run_dir).run()
with fixed time stepping (adaptivity FIXED); time-dependent boundary values
are imposed by mutating the channel's FluidComponentOperations from the
documented per-step callback (the production mechanism for boundary
schedules). Fluid: ACOUSTIC_FLUID (mu = 1e-5 Pa s), so the steady laminar
friction drop is 0.1% of the Joukowsky rise and line packing is negligible
against the 1-2% tolerances below.

Note (documented finding): an exactly zero initial mass-flow rate crashes
the initialization/assembly with NaN because the Blasius turbulent
correlation 0.316 Re^-0.25 has no zero-Reynolds guard (the laminar models
do); the eigenfrequency case therefore seeds the closed line with a
physically irrelevant 1e-12 kg/s.
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
STEADY_MASS_RATE = 5.0e-4  # kg/s: Re = 636 (laminar), Joukowsky rise 201 Pa
CLOSURE_TIME = 0.3  # s (initial transient of the consistent init has died)
WAVE_TRANSIT = 2.0 * CHANNEL_LENGTH / WAVE_SPEED  # 2L/a = 0.6325 s
WAVE_PERIOD = 4.0 * CHANNEL_LENGTH / WAVE_SPEED  # 4L/a = 1.2649 s
FLOW_SEED = 1.0e-12  # kg/s (see module docstring; Z * seed = 4e-7 Pa)


class OutletClosure:
    """Zero the imposed outlet mass rate from closure_time on: the outlet
    velocity boundary value drops to zero within one time step."""

    def __init__(self, closure_time):
        self.closure_time = closure_time

    def __call__(self, simulation):
        if simulation.simulation_time[-1] >= self.closure_time - 1.0e-9:
            channel_operations(simulation).outlet_mass_rate = 0.0


def closure_run(tmp_path, name, number_of_elements, time_step, end_time):
    """Steady laminar flow (imposed inlet pressure + outlet mass rate),
    instantaneous outlet closure at CLOSURE_TIME; returns the closed-end
    pressure trace (t, p)."""
    run_directory = write_channel_run_directory(
        tmp_path / name,
        number_of_elements=number_of_elements,
        time_step=time_step,
        end_time=end_time,
        method="BDF2",
        hydraulic_boundary_condition=2,
        inlet_pressure=BASE_PRESSURE,
        outlet_pressure=BASE_PRESSURE,
        initial_pressure=BASE_PRESSURE,
        inlet_mass_rate=STEADY_MASS_RATE,
        outlet_mass_rate=STEADY_MASS_RATE,
    )
    probe = ChannelProbe(node_index=-1)
    simulation = run_case(
        run_directory, step_callback=StepDriver(OutletClosure(CLOSURE_TIME), probe)
    )
    probe.record_final(simulation)
    times, pressures, _ = probe.as_arrays()
    return times, pressures


def pre_closure_pressure(times, pressures):
    window = (times > CLOSURE_TIME - 0.05) & (times <= CLOSURE_TIME)
    return pressures[window].mean()


def test_joukowsky_water_hammer(acoustic_fluid, tmp_path):
    # 401 elements, dt = 1e-3 s: CFL = a dt / h = 1.27, i.e. the mesh/step
    # pairing resolves the front as sharply as the implicit scheme allows.
    times, pressures = closure_run(
        tmp_path, "joukowsky", number_of_elements=401, time_step=1.0e-3,
        end_time=CLOSURE_TIME + 2.0 * WAVE_TRANSIT,
    )
    p_before = pre_closure_pressure(times, pressures)

    # Plateau value: median over the central half of the first 2L/a window,
    # away from the fronts. Tolerance 1%: the scheme's upwind dissipation
    # (numerical viscosity ~ (|v|+a) h / 2) smears the fronts over
    # ~ sqrt(2 L h) = 0.7 m but leaves the plateau level itself intact; the
    # residual measured bias is the ~0.1% line-packing tilt from laminar
    # friction plus BDF2 start-up damping (measured total +0.2%).
    window = (times > CLOSURE_TIME + 0.25 * WAVE_TRANSIT) & (
        times < CLOSURE_TIME + 0.75 * WAVE_TRANSIT
    )
    rise = np.median(pressures[window]) - p_before
    joukowsky = CHARACTERISTIC_IMPEDANCE * STEADY_MASS_RATE
    assert rise == pytest.approx(joukowsky, rel=1.0e-2)

    # Plateau duration between the half-amplitude crossings of the rising
    # and falling fronts (linear interpolation between samples). The
    # numerical front smear is symmetric about each ideal front, so the
    # mid-crossings are unbiased to first order; 2% covers the measured
    # -1.6% residual (one-step closure ramp + slightly asymmetric smear
    # growth between the two fronts).
    normalized = (pressures - p_before) / rise
    after = times >= CLOSURE_TIME - 2.0e-3
    t_after, s_after = times[after], normalized[after]
    rising = np.nonzero((s_after[:-1] < 0.5) & (s_after[1:] >= 0.5))[0][0]
    t_up = t_after[rising] + (0.5 - s_after[rising]) / (
        s_after[rising + 1] - s_after[rising]
    ) * (t_after[rising + 1] - t_after[rising])
    falling = np.nonzero((s_after[:-1] >= 0.5) & (s_after[1:] < 0.5))[0][0]
    t_down = t_after[falling] + (s_after[falling] - 0.5) / (
        s_after[falling] - s_after[falling + 1]
    ) * (t_after[falling + 1] - t_after[falling])
    assert t_down - t_up == pytest.approx(WAVE_TRANSIT, rel=2.0e-2)


def test_organ_pipe_eigenfrequencies(acoustic_fluid, tmp_path):
    # Quarter-wave resonator: pressure node at the inlet (imposed-pressure
    # boundary), antinode at the closed outlet; f_n = (2n-1) a / (4L).
    # Crank-Nicolson keeps the record dissipation-free in time (BDF2's
    # per-step damping would erase the third mode over the record); the
    # remaining mode damping is the spatial upwind dissipation, whose decay
    # rate ~ (a h / 2) k_n^2 sets the 30 s record: long enough for ~0.03 Hz
    # bin resolution, short enough that mode 3 (1/e time 8 s) still
    # dominates its spectral neighbourhood.
    frequencies = analytical.organ_pipe_frequencies(WAVE_SPEED, CHANNEL_LENGTH, 3)
    time_step = 1.0 / (40.0 * frequencies[2])  # 40 samples per f_3 period
    pulse_start, pulse_width, pulse_amplitude = 0.05, 0.1, 1.0e3
    number_of_elements = 401

    run_directory = write_channel_run_directory(
        tmp_path / "organ_pipe",
        number_of_elements=number_of_elements,
        time_step=time_step,
        end_time=30.0,
        method="CN",
        hydraulic_boundary_condition=2,
        inlet_pressure=BASE_PRESSURE,
        outlet_pressure=BASE_PRESSURE,
        initial_pressure=BASE_PRESSURE,
        inlet_mass_rate=FLOW_SEED,
        outlet_mass_rate=FLOW_SEED,
    )

    def inlet_pulse(simulation):
        # Boundary values apply to the upcoming time level t + dt.
        t_next = simulation.simulation_time[-1] + time_step
        channel_operations(simulation).inlet_pressure = BASE_PRESSURE + float(
            analytical.raised_cosine_pulse(
                np.asarray(t_next), pulse_start, pulse_width, pulse_amplitude
            )
        )

    probe = ChannelProbe(node_index=number_of_elements // 2)  # midpoint: all
    # three modes have |sin((2n-1) pi/4)| = 0.707 there — none is at a node.
    simulation = run_case(
        run_directory, step_callback=StepDriver(inlet_pulse, probe)
    )
    probe.record_final(simulation)
    times, pressures, _ = probe.as_arrays()

    window = times >= pulse_start + pulse_width + 0.1
    measured = analytical.fft_peak_frequencies(pressures[window], time_step, 3)
    # 1% per peak: the Hann + zero-padding + parabolic-interpolation
    # estimator resolves ~0.3 of the 0.033 Hz bin (~0.5% of f_1 worst case),
    # and the CN phase error at f_3 with 40 samples/period is 0.2%
    # (measured errors: 0.005%, 0.07%, 0.21%).
    np.testing.assert_allclose(measured, frequencies, rtol=1.0e-2)


def test_first_mode_mesh_convergence(acoustic_fluid, tmp_path):
    # Mesh-convergence evidence for the wave dynamics: the closed-end
    # pressure after instantaneous closure is exactly a square wave of
    # amplitude Z mdot0 and period 4L/a (the superposition of all odd
    # organ-pipe modes). The relative L2 distance of the simulated trace
    # from that exact solution over the first two periods is dominated by
    # the upwind front smearing ~ sqrt(a t h), so it must decrease strictly
    # under mesh refinement at fixed dt (measured: 0.293, 0.247, 0.209,
    # 0.182 for 51/101/201/401 elements). The eigenfrequency *peak* error
    # itself (~1e-4 at 51 elements) sits far below the FFT estimator
    # resolution and cannot demonstrate convergence honestly.
    time_step = 2.0e-3
    errors = []
    for number_of_elements in (51, 101, 201, 401):
        times, pressures = closure_run(
            tmp_path,
            f"mesh_{number_of_elements}",
            number_of_elements=number_of_elements,
            time_step=time_step,
            end_time=CLOSURE_TIME + 2.0 * WAVE_PERIOD + 0.05,
        )
        p_before = pre_closure_pressure(times, pressures)
        window = (times > CLOSURE_TIME) & (times <= CLOSURE_TIME + 2.0 * WAVE_PERIOD)
        reference = analytical.closed_end_square_wave(
            times[window] - CLOSURE_TIME,
            p_before,
            CHARACTERISTIC_IMPEDANCE * STEADY_MASS_RATE,
            WAVE_PERIOD,
        )
        errors.append(
            np.linalg.norm(pressures[window] - reference)
            / np.linalg.norm(reference - p_before)
        )
    assert np.all(np.diff(errors) < 0.0), f"errors not decreasing: {errors}"
