"""One-off: build ``validation_data/aab51`` from the AAB51 measurement archive.

Copies the fast event files of the sensors used by the models and figures
unchanged, and the slow-archive files of the same sensors restricted to a
time window around the quench (header lines kept, every other byte of the
kept lines untouched: Latin-1, CRLF endings). Writes ``PROVENANCE.md`` with
the file list and the window, and checks that the package reader returns
the same arrays as on the archive within the window.

Usage (from the repository root):
    .venv/bin/python models/w7x/common/tools/trim_measurement_data.py \
        --source W7-X/quench_test_validation/measurement_data \
        --target models/w7x/validation_data/aab51 --window -4500 300
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from common.constants import QUENCH_TIMESTAMP  # noqa: E402

# Sensors of the fast event files (Quench/), by role.
EVENT_SENSORS = {
    "DUP11-3": "compensated detection voltage (1,2)-(3,4), figures",
    "IAL-C1": "coil current, dump waveform of the controller",
    "TT2100A": "outlet temperature double layer (1,2), figures",
    "TT2101A": "outlet temperature double layers (3,4)+(5,6), figures",
    "TT2102A": "outlet temperature double layers (7,8)+(9,10), figures",
    "TT2103A": "outlet temperature double layer (11,12), figures",
    "PT2111": "supply pressure, drives the supply reservoir",
    "TT2113C": "supply temperature, drives the supply reservoir",
}
# Slow-archive files (Slow/), trimmed to the window.
SLOW_SENSORS = (
    "DUP11-3_ai", "TT2100A_ai", "TT2101A_ai", "TT2102A_ai", "TT2103A_ai",
    "PT2111_ai", "TT2113C_ai",
)


def event_file(source: Path, sensor: str) -> Path:
    matches = sorted(source.glob(f"Quench/[qQ]*_C1_{sensor}_ai.txt"))
    if not matches:
        raise FileNotFoundError(sensor)
    return matches[0]  # uppercase (0.2 ms) sorts first when it exists


def slow_time(line: bytes):
    """Measured time (s) of a slow-archive data line, None for other lines."""
    parts = line.strip().split()
    if len(parts) != 3 or parts[0].startswith((b"#", b"00/00")):
        return None
    try:
        stamp = datetime.strptime(
            f"{parts[0].decode('latin-1')} {parts[1].decode('latin-1')[:15]}",
            "%d/%m/%Y %H:%M:%S.%f",
        )
    except ValueError:
        return None
    return (stamp - QUENCH_TIMESTAMP).total_seconds()


def trim_slow_file(source: Path, target: Path, window) -> tuple[int, int]:
    raw = source.read_bytes()
    lines = raw.split(b"\n")
    kept, total = [], 0
    for line in lines:
        time = slow_time(line)
        if time is None:
            if line.strip() == b"" or line.lstrip().startswith(b"#"):
                kept.append(line)
            continue
        total += 1
        if window[0] <= time <= window[1]:
            kept.append(line)
    target.write_bytes(b"\n".join(kept))
    return total, sum(1 for line in kept if slow_time(line) is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--window", type=float, nargs=2, default=(-4500.0, 300.0))
    arguments = parser.parse_args()
    source, target, window = arguments.source, arguments.target, arguments.window
    (target / "Quench").mkdir(parents=True, exist_ok=True)
    (target / "Slow").mkdir(parents=True, exist_ok=True)

    rows = []
    for sensor, role in EVENT_SENSORS.items():
        path = event_file(source, sensor)
        shutil.copy2(path, target / "Quench" / path.name)
        rows.append(f"| `Quench/{path.name}` | {role} | complete |")
    for sensor in SLOW_SENSORS:
        total, kept = trim_slow_file(
            source / "Slow" / f"{sensor}.txt", target / "Slow" / f"{sensor}.txt",
            window,
        )
        rows.append(f"| `Slow/{sensor}.txt` | slow archive of {sensor[:-3]} | "
                    f"{kept} of {total} rows |")
    shutil.copy2(source / "readme.txt", target / "readme.txt")

    # Self-check with the package reader.
    import os

    from common import measurement_data as md

    os.environ["W7X_VALIDATION_DATA"] = str(target.parent)
    md.MEASUREMENT_DIRECTORY = target
    archive = source
    for sensor in SLOW_SENSORS:
        md.MEASUREMENT_DIRECTORY = archive
        time_full, value_full = md.read_slow_signal(sensor)
        mask = (time_full >= window[0]) & (time_full <= window[1])
        md.MEASUREMENT_DIRECTORY = target
        time_trim, value_trim = md.read_slow_signal(sensor)
        assert np.array_equal(time_full[mask], time_trim), sensor
        assert np.array_equal(value_full[mask], value_trim), sensor
    for sensor in EVENT_SENSORS:
        md.MEASUREMENT_DIRECTORY = archive
        full = md.read_event_signal(sensor)
        md.MEASUREMENT_DIRECTORY = target
        trimmed = md.read_event_signal(sensor)
        assert np.array_equal(full[0], trimmed[0]) and np.array_equal(full[1], trimmed[1])
    print("self-check passed: package reader identical to the archive in the window")

    provenance = "\n".join(
        [
            "# AAB51 measurement subset",
            "",
            "Quench test of W7-X coil AAB51 at the CEA Saclay test facility, event of",
            "2007-03-06 14:51:53.57 (fast-acquisition trigger = measured time 0).",
            f"Built from the measurement archive `{source}` with",
            "`common/tools/trim_measurement_data.py`.",
            "",
            "* `Quench/`: fast event files, window [-1.43, +8.57] s around the trigger,",
            "  numeric TIME column; uppercase `Q` files 0.2 ms, lowercase `q` 10 ms;",
            "  Latin-1 encoded, copied unchanged.",
            f"* `Slow/`: slow control archive ('dd/mm/yyyy hh:mm:ss.f value' rows),",
            f"  restricted to measured time [{window[0]:g}, {window[1]:g}] s; header lines",
            "  and the kept rows are byte-identical to the archive.",
            "* `readme.txt`: the sensor list of the archive (covers more sensors than",
            "  packaged here).",
            "",
            "| file | role | content |",
            "|---|---|---|",
            *rows,
            "",
            "Only these sensors are used: PT2111 / TT2113C drive the supply reservoir of",
            "the winding-pack runs, IAL-C1 provides the dump waveform, DUP11-3 and",
            "TT2100A-TT2103A are compared with the simulations in the figures.",
            "",
        ]
    )
    (target / "PROVENANCE.md").write_text(provenance)
    print(f"wrote {target / 'PROVENANCE.md'}")


if __name__ == "__main__":
    main()
