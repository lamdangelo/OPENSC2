"""V5 figure: node thermal first-order lag vs analytics, with the pressure
invariance residual.

Run with the repo venv:  .venv/bin/python scripts/verification/run_v5_thermal_lag.py
"""

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.coolprop_interface as cpi  # noqa: E402

import analytical  # noqa: E402
import test_v5_thermal_lag as v5  # noqa: E402
from case_builders import VERIFICATION_FLUID  # noqa: E402
from _bootstrap import save_figure  # noqa: E402


def main():
    cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    time_step = 2.5e-3 / v5.RATE
    number_of_steps = int(np.ceil(5.0 / v5.RATE / time_step))
    network = v5.lag_network()
    times, temperatures, pressures, _ = v5.run_lag(
        network, time_step, number_of_steps
    )
    reference = analytical.first_order_lag(
        times, v5.INITIAL_TEMPERATURE, v5.INFLOW_TEMPERATURE, v5.RATE
    )

    figure, axes = plt.subplots(
        3, 1, figsize=(6.4, 5.4), constrained_layout=True,
        sharex=True, height_ratios=[2.2, 1, 1],
    )
    axes[0].plot(times, temperatures, color="tab:blue", label="solver (BE)")
    axes[0].plot(times, reference, color="tab:orange", linestyle="--",
                 label="analytical")
    axes[0].set_ylabel("node T [K]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        rf"V5: first-order thermal lag ($\tau$ = {1.0 / v5.RATE:.0f} s, "
        rf"$\lambda\,$dt = {v5.RATE * time_step:.4g})"
    )

    axes[1].plot(times, np.abs(temperatures - reference), color="tab:blue")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|T error| [K]")
    axes[1].grid(alpha=0.3)

    pressure_residual = np.abs(pressures - v5.NODE_PRESSURE) / v5.NODE_PRESSURE
    axes[2].plot(times, np.maximum(pressure_residual, 1e-18), color="tab:green")
    axes[2].set_yscale("log")
    axes[2].axhline(1e-10, color="tab:red", linewidth=1, linestyle="--")
    axes[2].annotate("test tolerance", (0.02, 0.75), xycoords="axes fraction",
                     color="tab:red", fontsize=8)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("|p residual| / p")
    axes[2].grid(alpha=0.3)
    save_figure(figure, "v5_thermal_lag")
    plt.close(figure)


if __name__ == "__main__":
    main()
