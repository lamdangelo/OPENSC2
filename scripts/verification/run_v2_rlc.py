"""V2 figures: surge-tank RLC free response vs analytics, and the
time-integrator convergence study.

Run with the repo venv:  .venv/bin/python scripts/verification/run_v2_rlc.py
"""

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.coolprop_interface as cpi  # noqa: E402
from conductor.conductor_flags import MethodFlag  # noqa: E402

import analytical  # noqa: E402
import test_v2_rlc as v2  # noqa: E402
from case_builders import VERIFICATION_FLUID  # noqa: E402
from _bootstrap import save_figure  # noqa: E402

METHODS = [
    (MethodFlag.BACKWARD_EULER, "backward Euler", 1.0, "tab:blue"),
    (MethodFlag.CRANK_NICOLSON, "Crank-Nicolson", 2.0, "tab:orange"),
    (MethodFlag.GALERKIN, "Galerkin (theta=2/3)", 1.0, "tab:green"),
    (MethodFlag.BACKWARD_DIFFERENCE_2, "BDF2", 2.0, "tab:red"),
]


def free_response_figure():
    time_step = 0.01 / v2.OMEGA_D
    number_of_steps = int(np.ceil(8.0 * v2.DAMPED_PERIOD / time_step))
    network = v2.rlc_network(MethodFlag.BACKWARD_DIFFERENCE_2)
    times, pressures = v2.run_free_response(network, time_step, number_of_steps)
    simulated = pressures - v2.RESERVOIR_PRESSURE
    reference = v2.analytical_offset(times)
    envelope = v2.PRESSURE_OFFSET * np.exp(-v2.SIGMA * times) / np.cos(
        np.arctan(v2.SIGMA / v2.OMEGA_D)
    )

    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), constrained_layout=True,
        sharex=True, height_ratios=[2.2, 1],
    )
    axis_top.plot(times, simulated, color="tab:blue", label="solver (BDF2)")
    axis_top.plot(
        times, reference, color="tab:orange", linestyle="--",
        label="analytical",
    )
    axis_top.plot(times, envelope, color="tab:gray", linewidth=1,
                  linestyle=":", label="analytical envelope")
    axis_top.plot(times, -envelope, color="tab:gray", linewidth=1,
                  linestyle=":")
    axis_top.set_ylabel(r"$p - p_\mathrm{res}$ (Pa)")
    axis_top.legend(loc="upper right", fontsize=8)
    axis_top.grid(alpha=0.3)
    axis_top.set_title(
        "V2: surge-tank RLC free response "
        rf"($\sigma/\omega_0$ = {v2.DAMPING_RATIO}, $\omega_d\,$dt = 0.01)"
    )

    axis_bottom.plot(times, np.abs(simulated - reference), color="tab:blue")
    axis_bottom.set_yscale("log")
    axis_bottom.set_xlabel("time (s)")
    axis_bottom.set_ylabel("|error| (Pa)")
    axis_bottom.grid(alpha=0.3)
    save_figure(figure, "v2_rlc_free_response")
    plt.close(figure)


def convergence_figure():
    base_step = v2.DAMPED_PERIOD / 100.0
    step_sizes = base_step / 2 ** np.arange(4)
    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for method, label, expected_order, color in METHODS:
        errors = []
        for time_step in step_sizes:
            number_of_steps = int(np.round(2.0 * v2.DAMPED_PERIOD / time_step))
            network = v2.rlc_network(method)
            times, pressures = v2.run_free_response(
                network, time_step, number_of_steps
            )
            errors.append(
                analytical.relative_l2_error(
                    pressures - v2.RESERVOIR_PRESSURE,
                    v2.analytical_offset(times),
                )
            )
        order = analytical.fitted_order(step_sizes, np.asarray(errors))
        axis.loglog(
            v2.OMEGA_D * step_sizes, errors, "o-", color=color,
            label=f"{label}: order {order:.2f} (theory {expected_order:.0f})",
        )
    reference_x = v2.OMEGA_D * step_sizes
    for slope, anchor in ((1.0, 0.55), (2.0, 0.1)):
        axis.loglog(
            reference_x,
            anchor * (reference_x / reference_x[0]) ** slope,
            color="tab:gray", linewidth=1, linestyle="--",
        )
        axis.annotate(
            f"slope {slope:.0f}", (reference_x[-1], anchor
            * (reference_x[-1] / reference_x[0]) ** slope),
            fontsize=8, color="tab:gray",
        )
    axis.set_xlabel(r"$\omega_d\,$dt")
    axis.set_ylabel("relative L2 error over 2 periods")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    axis.set_title("V2: time-integrator convergence")
    save_figure(figure, "v2_rlc_convergence")
    plt.close(figure)


def main():
    cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    free_response_figure()
    convergence_figure()


if __name__ == "__main__":
    main()
