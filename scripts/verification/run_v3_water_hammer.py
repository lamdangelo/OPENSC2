"""V3 figures: Joukowsky water hammer, organ-pipe spectrum, and mesh
convergence of the closed-end pressure trace. Reruns the channel cases in a
temporary directory (a few minutes).

Run with the repo venv:  .venv/bin/python scripts/verification/run_v3_water_hammer.py
"""

import _bootstrap  # noqa: F401

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.coolprop_interface as cpi  # noqa: E402

import analytical  # noqa: E402
import test_v3_water_hammer as v3  # noqa: E402
from case_builders import (  # noqa: E402
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
from _bootstrap import save_figure  # noqa: E402


def joukowsky_figure(work_directory):
    times, pressures = v3.closure_run(
        work_directory, "joukowsky", number_of_elements=401, time_step=1.0e-3,
        end_time=v3.CLOSURE_TIME + 2.0 * v3.WAVE_PERIOD,
    )
    p_before = v3.pre_closure_pressure(times, pressures)
    reference = np.where(
        times <= v3.CLOSURE_TIME,
        p_before,
        analytical.closed_end_square_wave(
            times - v3.CLOSURE_TIME,
            p_before,
            CHARACTERISTIC_IMPEDANCE * v3.STEADY_MASS_RATE,
            v3.WAVE_PERIOD,
        ),
    )

    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), constrained_layout=True,
        sharex=True, height_ratios=[2.2, 1],
    )
    axis_top.plot(times, pressures - p_before, color="tab:blue",
                  label="solver (BDF2, 401 elements)")
    axis_top.plot(times, reference - p_before, color="tab:orange",
                  linestyle="--", label="exact square wave")
    axis_top.axhline(
        CHARACTERISTIC_IMPEDANCE * v3.STEADY_MASS_RATE, color="tab:gray",
        linewidth=1, linestyle=":",
    )
    axis_top.annotate(
        r"Joukowsky rise $Z\,\dot m_0$",
        (0.02, 0.9), xycoords="axes fraction", fontsize=8, color="tab:gray",
    )
    axis_top.set_ylabel("p - p0 at closed end [Pa]")
    axis_top.legend(loc="lower left", fontsize=8)
    axis_top.grid(alpha=0.3)
    axis_top.set_title("V3a: water hammer after instantaneous outlet closure")

    axis_bottom.plot(times, np.abs(pressures - reference), color="tab:blue")
    axis_bottom.set_yscale("log")
    axis_bottom.set_xlabel("time [s]")
    axis_bottom.set_ylabel("|error| [Pa]")
    axis_bottom.grid(alpha=0.3)
    save_figure(figure, "v3a_joukowsky")
    plt.close(figure)


def spectrum_figure(work_directory):
    frequencies = analytical.organ_pipe_frequencies(WAVE_SPEED, CHANNEL_LENGTH, 3)
    time_step = 1.0 / (40.0 * frequencies[2])
    pulse_start, pulse_width, pulse_amplitude = 0.05, 0.1, 1.0e3
    run_directory = write_channel_run_directory(
        Path(work_directory) / "organ_pipe",
        number_of_elements=401, time_step=time_step, end_time=30.0,
        method="CN", hydraulic_boundary_condition=2,
        inlet_pressure=v3.BASE_PRESSURE, outlet_pressure=v3.BASE_PRESSURE,
        initial_pressure=v3.BASE_PRESSURE,
        inlet_mass_rate=v3.FLOW_SEED, outlet_mass_rate=v3.FLOW_SEED,
    )

    def inlet_pulse(simulation):
        t_next = simulation.simulation_time[-1] + time_step
        channel_operations(simulation).inlet_pressure = v3.BASE_PRESSURE + float(
            analytical.raised_cosine_pulse(
                np.asarray(t_next), pulse_start, pulse_width, pulse_amplitude
            )
        )

    probe = ChannelProbe(node_index=200)
    simulation = run_case(run_directory, step_callback=StepDriver(inlet_pulse, probe))
    probe.record_final(simulation)
    times, pressures, _ = probe.as_arrays()
    window = times >= pulse_start + pulse_width + 0.1
    samples = pressures[window] - pressures[window].mean()
    padded = 8 * samples.size
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size), n=padded))
    spectral_frequencies = np.fft.rfftfreq(padded, d=time_step)
    measured = analytical.fft_peak_frequencies(pressures[window], time_step, 3)

    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), constrained_layout=True,
        height_ratios=[2.2, 1],
    )
    band = spectral_frequencies <= 5.0
    axis_top.semilogy(
        spectral_frequencies[band], spectrum[band] / spectrum.max(),
        color="tab:blue", label="midpoint pressure spectrum",
    )
    for n, frequency in enumerate(frequencies, start=1):
        axis_top.axvline(frequency, color="tab:orange", linewidth=1,
                         linestyle="--")
        axis_top.annotate(
            rf"$f_{n}$", (frequency, 1.2), fontsize=8, color="tab:orange",
            ha="center", annotation_clip=False,
        )
    axis_top.set_xlabel("frequency [Hz]")
    axis_top.set_ylabel("normalized amplitude")
    axis_top.legend(loc="lower left", fontsize=8)
    axis_top.grid(alpha=0.3)
    axis_top.set_title("V3b: organ-pipe eigenfrequencies (CN, 401 elements)")

    relative_errors = np.abs(measured / frequencies - 1.0)
    axis_bottom.bar([1, 2, 3], relative_errors, 0.5, color="tab:gray")
    axis_bottom.axhline(1e-2, color="tab:red", linewidth=1, linestyle="--")
    axis_bottom.annotate("test tolerance", (0.02, 0.8),
                         xycoords="axes fraction", color="tab:red", fontsize=8)
    axis_bottom.set_yscale("log")
    axis_bottom.set_xticks([1, 2, 3], [r"$f_1$", r"$f_2$", r"$f_3$"])
    axis_bottom.set_ylabel("|peak error| (rel.)")
    axis_bottom.grid(alpha=0.3)
    save_figure(figure, "v3b_eigenfrequencies")
    plt.close(figure)


def mesh_convergence_figure(work_directory):
    meshes = (51, 101, 201, 401)
    errors = []
    for number_of_elements in meshes:
        times, pressures = v3.closure_run(
            work_directory, f"mesh_{number_of_elements}",
            number_of_elements=number_of_elements, time_step=2.0e-3,
            end_time=v3.CLOSURE_TIME + 2.0 * v3.WAVE_PERIOD + 0.05,
        )
        p_before = v3.pre_closure_pressure(times, pressures)
        window = (times > v3.CLOSURE_TIME) & (
            times <= v3.CLOSURE_TIME + 2.0 * v3.WAVE_PERIOD
        )
        reference = analytical.closed_end_square_wave(
            times[window] - v3.CLOSURE_TIME, p_before,
            CHARACTERISTIC_IMPEDANCE * v3.STEADY_MASS_RATE, v3.WAVE_PERIOD,
        )
        errors.append(
            np.linalg.norm(pressures[window] - reference)
            / np.linalg.norm(reference - p_before)
        )

    element_sizes = CHANNEL_LENGTH / np.asarray(meshes)
    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    axis.loglog(element_sizes, errors, "o-", color="tab:blue",
                label="relative L2 error vs exact square wave")
    axis.loglog(
        element_sizes,
        errors[-1] * (element_sizes / element_sizes[-1]) ** 0.5,
        color="tab:gray", linewidth=1, linestyle="--",
        label=r"slope 1/2 (front smear $\propto \sqrt{h}$)",
    )
    axis.set_xlabel("element size h [m]")
    axis.set_ylabel("relative L2 error over 2 periods")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    axis.set_title("V3: mesh convergence of the closed-end pressure trace")
    save_figure(figure, "v3c_mesh_convergence")
    plt.close(figure)


def main():
    cpi.set_constant_fluid_properties(ACOUSTIC_FLUID)
    with tempfile.TemporaryDirectory() as work_directory:
        work_directory = Path(work_directory)
        joukowsky_figure(work_directory)
        spectrum_figure(work_directory)
        mesh_convergence_figure(work_directory)


if __name__ == "__main__":
    main()
