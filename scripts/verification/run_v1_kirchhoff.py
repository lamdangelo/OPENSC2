"""V1 figure: Wheatstone-bridge steady state, solver vs Kirchhoff reference.

Run with the repo venv:  .venv/bin/python scripts/verification/run_v1_kirchhoff.py
"""

import _bootstrap  # noqa: F401  (backend + sys.path side effects)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.coolprop_interface as cpi  # noqa: E402

import analytical  # noqa: E402
import test_v1_kirchhoff as v1  # noqa: E402
from case_builders import VERIFICATION_FLUID  # noqa: E402
from _bootstrap import save_figure  # noqa: E402


def main():
    cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    network = v1.wheatstone_network()
    network.solve_steady_state()

    resistances = np.array(
        [
            analytical.laminar_pipe_resistance(
                VERIFICATION_FLUID.viscosity,
                length,
                v1.DIAMETER,
                VERIFICATION_FLUID.density,
                v1.CROSS_SECTION,
            )
            for (_, _, length) in v1.BRANCHES.values()
        ]
    )
    internal_incidence, reservoir_incidence = v1.incidence_matrices()
    _, reference_flows = analytical.steady_network_reference(
        internal_incidence,
        reservoir_incidence,
        resistances,
        np.array([v1.SUPPLY_PRESSURE, v1.RETURN_PRESSURE]),
    )
    solver_flows = np.array(
        [network.branch_mass_flow_of(identifier) for identifier in v1.BRANCHES]
    )

    labels = list(v1.BRANCHES)
    positions = np.arange(len(labels))
    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.5), constrained_layout=True,
        sharex=True, height_ratios=[2.2, 1],
    )
    width = 0.38
    axis_top.bar(
        positions - width / 2, 1e3 * solver_flows, width,
        label="solver", color="tab:blue",
    )
    axis_top.bar(
        positions + width / 2, 1e3 * reference_flows, width,
        label="Kirchhoff reference", color="tab:orange",
    )
    axis_top.set_ylabel("mass flow [g/s]")
    axis_top.legend()
    axis_top.grid(alpha=0.3)
    axis_top.set_title("V1: Wheatstone bridge, steady branch flows")

    residual = np.abs(solver_flows - reference_flows) / np.max(
        np.abs(reference_flows)
    )
    axis_bottom.bar(positions, residual, 0.5, color="tab:gray")
    axis_bottom.set_yscale("log")
    axis_bottom.set_ylabel("|error| / max|flow|")
    axis_bottom.set_xticks(positions, labels, rotation=30)
    axis_bottom.axhline(1e-8, color="tab:red", linewidth=1, linestyle="--")
    axis_bottom.annotate(
        "test tolerance", (0.02, 0.8), xycoords="axes fraction",
        color="tab:red", fontsize=8,
    )
    axis_bottom.grid(alpha=0.3)
    save_figure(figure, "v1_kirchhoff")
    plt.close(figure)


if __name__ == "__main__":
    main()
