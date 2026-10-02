"""Hot-spot comparison of the adiabatic 1D model with the THEA reference.

Reads the OPENSC2 strand temperature at the heated zone (time-evolution
probe at z = 80.24 m of simulation_results/W7X) and the THEA strand hot-spot
temperature (validation_data/thea/high_fidelity_data_thermal.csv), computes
the relative difference |T_THEA - T_OPENSC2| / T_THEA on the OPENSC2 time
grid (THEA linearly interpolated; the comparison covers the common time
span, 0-10 s for the packaged deck) and writes

    figures/hot_spot_comparison_thea_opensc.eps
    figures/relative_difference_thea_opensc.csv

Usage (after run_model.py):
    .venv/bin/python models/w7x/adiabatic_model/make_figures.py [--model-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_DIRECTORY = Path(__file__).resolve().parent
THEA_FILE = (
    MODEL_DIRECTORY.parent / "validation_data" / "thea" / "high_fidelity_data_thermal.csv"
)
PROBE_POSITION = 80.24  # m, time-evolution probe inside the heated zone

# MPG corporate design colours
MPG_GREEN = (0 / 255, 108 / 255, 102 / 255)
MPG_ORANGE = (239 / 255, 124 / 255, 0 / 255)


def opensc_hot_spot(model_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    matches = sorted(
        model_directory.glob(
            "simulation_results/*/Output/Time_evolution/CONDUCTOR_1/"
            "STR_MIX_1_temperature_te.tsv"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"no STR_MIX_1_temperature_te.tsv under {model_directory}; run "
            "run_model.py first"
        )
    frame = pd.read_csv(matches[-1], sep="\t")
    probes = {
        column: float(column.split("=")[1].split("(")[0])
        for column in frame.columns
        if column.startswith("zcoord")
    }
    column = min(probes, key=lambda name: abs(probes[name] - PROBE_POSITION))
    return frame["time (s)"].to_numpy(), frame[column].to_numpy()


def thea_hot_spot() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(THEA_FILE)
    return frame["Time"].to_numpy(), frame["Strand Temperature"].to_numpy()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY)
    arguments = parser.parse_args(argv)
    output_directory = arguments.model_dir / "figures"
    output_directory.mkdir(exist_ok=True)

    time_opensc, temperature_opensc = opensc_hot_spot(arguments.model_dir)
    time_thea, temperature_thea = thea_hot_spot()
    common = time_opensc <= min(time_opensc[-1], time_thea[-1])
    thea_on_opensc = np.interp(time_opensc[common], time_thea, temperature_thea)
    relative_difference = (
        np.abs(thea_on_opensc - temperature_opensc[common]) / thea_on_opensc
    )
    pd.DataFrame(
        {"time (s)": time_opensc[common], "relative difference (-)": relative_difference}
    ).to_csv(output_directory / "relative_difference_thea_opensc.csv", index=False)

    fig, ax_temperature = plt.subplots(figsize=(6.4, 3.2))
    plt.grid()
    ax_error = ax_temperature.twinx()
    line_thea, = ax_temperature.plot(
        time_thea, temperature_thea, color=MPG_ORANGE, linestyle="-", label="THEA"
    )
    line_opensc, = ax_temperature.plot(
        time_opensc, temperature_opensc, color=MPG_GREEN, linestyle="--",
        label="OPENSC$^2$",
    )
    line_error, = ax_error.semilogy(
        time_opensc[common], relative_difference, color="black",
        linestyle="dotted", label="Rel. diff.",
    )
    ax_temperature.set_xlabel("Time (s)")
    ax_temperature.set_ylabel("Hot-spot temperature (K)")
    ax_error.set_ylabel("Relative difference (-)")
    lines = [line_thea, line_opensc, line_error]
    # EPS does not support transparency, hence the opaque legend frame
    ax_temperature.legend(lines, [line.get_label() for line in lines], framealpha=1)
    fig.tight_layout()
    target = output_directory / "hot_spot_comparison_thea_opensc.eps"
    fig.savefig(target, format="eps")
    plt.close(fig)
    print(
        f"[figure] {target} (peak OPENSC2 {temperature_opensc.max():.1f} K, "
        f"THEA {temperature_thea.max():.1f} K; max relative difference "
        f"{relative_difference.max():.3f})"
    )


if __name__ == "__main__":
    main()
