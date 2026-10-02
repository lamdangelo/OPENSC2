"""V4 figures: coupled transmission-line termination — probe traces for the
four terminations and the measured reflection coefficient against the
analytical Gamma(R/Z) curve. Reruns the coupled cases in a temporary
directory (a few minutes).

Run with the repo venv:  .venv/bin/python scripts/verification/run_v4_transmission_line.py
"""

import _bootstrap  # noqa: F401

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.coolprop_interface as cpi  # noqa: E402

import analytical  # noqa: E402
import test_v4_transmission_line as v4  # noqa: E402
from case_builders import ACOUSTIC_FLUID, CHARACTERISTIC_IMPEDANCE  # noqa: E402
from _bootstrap import save_figure  # noqa: E402

TERMINATIONS = [
    (r"matched ($Z_\mathrm{L} = -Z_\mathrm{C})$", CHARACTERISTIC_IMPEDANCE, "tab:blue"),
    (r"closed ($Z_\mathrm{L} = 10^{12}$)", v4.BLOCKED_RESISTANCE, "tab:orange"),
    (r"reservoir ($Z_\mathrm{L} = 10^{-4} Z_\mathrm{C}$)", 1.0e-4 * CHARACTERISTIC_IMPEDANCE, "tab:green"),
    (r"$Z_\mathrm{L} = -3 Z_\mathrm{C}$", 3.0 * CHARACTERISTIC_IMPEDANCE, "tab:red"),
]


def main():
    cpi.set_constant_fluid_properties(ACOUSTIC_FLUID)
    results = []
    with tempfile.TemporaryDirectory() as work_directory:
        work_directory = Path(work_directory)
        for index, (label, resistance, color) in enumerate(TERMINATIONS):
            times, pressures, baseline, incident, reflected = v4.run_termination(
                work_directory, f"case_{index}", resistance
            )
            gamma = v4.measured_gamma(
                times, pressures, baseline, incident, reflected
            )
            results.append(
                (label, resistance, color, times, pressures - baseline,
                 incident, reflected, gamma)
            )

    # Probe traces.
    figure, axes = plt.subplots(
        len(results), 1, figsize=(6.4, 6.4), constrained_layout=True,
        sharex=True,
    )
    for axis, (label, _, color, times, offset, incident, reflected,
               gamma) in zip(axes, results):
        axis.plot(times, offset, color=color, label=label)
        for window, window_label in ((incident, "incident"),
                                     (reflected, "reflected")):
            axis.axvspan(times[window][0], times[window][-1], color="tab:gray",
                         alpha=0.15)
        axis.annotate(rf"$\Gamma$ = {gamma:+.3f}", (0.99, 0.75),
                      xycoords="axes fraction", ha="right", fontsize=8)
        axis.legend(loc="upper left", fontsize=8)
        axis.grid(alpha=0.3)
        axis.set_ylabel(r"$p - p_0$ (Pa)")
    axes[0].set_title(
        "V4: pulse at the 3L/4 probe (shaded: incident / reflected windows)"
    )
    axes[-1].set_xlabel("time (s)")
    save_figure(figure, "v4_probe_traces")
    plt.close(figure)

    # Gamma vs R/Z.
    ratios = np.logspace(-4.2, 6.5, 300)
    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), constrained_layout=True,
        sharex=True, height_ratios=[2.2, 1],
    )
    axis_top.semilogx(
        ratios,
        analytical.reflection_coefficient(
            ratios * CHARACTERISTIC_IMPEDANCE, CHARACTERISTIC_IMPEDANCE
        ),
        color="tab:gray", label=r"$\Gamma = (Z_\mathrm{L}+Z_\mathrm{C})/(Z_\mathrm{L}-Z_\mathrm{C})$",
    )
    for label, resistance, color, *_, gamma in results:
        axis_top.semilogx(
            resistance / CHARACTERISTIC_IMPEDANCE, gamma, "o", color=color,
            label=label,
        )
    axis_top.set_ylabel(r"$\Gamma$")
    axis_top.legend(fontsize=8)
    axis_top.grid(alpha=0.3)
    axis_top.set_title("V4: reflection coefficient of the lumped termination")

    for label, resistance, color, *_, gamma in results:
        expected = analytical.reflection_coefficient(
            resistance, CHARACTERISTIC_IMPEDANCE
        )
        axis_bottom.semilogx(
            resistance / CHARACTERISTIC_IMPEDANCE, abs(gamma - expected), "o",
            color=color,
        )
    axis_bottom.axhline(0.02, color="tab:red", linewidth=1, linestyle="--")
    axis_bottom.annotate("test tolerance", (0.02, 0.75),
                         xycoords="axes fraction", color="tab:red", fontsize=8)
    axis_bottom.set_yscale("log")
    axis_bottom.set_xlabel("R / Z")
    axis_bottom.set_ylabel(r"$|\Gamma - \Gamma_{exact}|$")
    axis_bottom.grid(alpha=0.3)
    save_figure(figure, "v4_reflection_coefficient")
    plt.close(figure)


if __name__ == "__main__":
    main()
