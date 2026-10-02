"""Readers for the Saclay AAB51 measurement archive.

Two file families (see readme.txt):
    * Quench/ - event window [-1.43 s, +8.57 s] around the quench
      timestamp (2007-03-06 14:51:53.57); numeric TIME column; lowercase
      files 10 ms, uppercase 0.2 ms; Latin-1 encoded.
    * Slow/   - slow control archive over weeks; "dd/mm/yyyy hh:mm:ss.f
      value" rows; 40 s sampling (coil current 1 s, U11 2 s); rows with
      #N/A or zeroed dates are invalid, FI values of -1.0 are sensor
      dropouts.

All times returned are measured time: seconds relative to the quench
timestamp (negative = before the quench). The reader logic mirrors the
user's Sensor_Data_Reader (quench-model-benchmark-study).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

MEASUREMENT_DIRECTORY = Path(
    os.environ.get(
        "W7X_VALIDATION_DATA",
        Path(__file__).resolve().parents[1] / "validation_data",
    )
) / "aab51"
QUENCH_TIMESTAMP = datetime(2007, 3, 6, 14, 51, 53, 573894)


def read_event_signal(sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """Fast/slow event file from Quench/ by sensor stem, e.g. 'TT2113C'."""
    matches = sorted(
        MEASUREMENT_DIRECTORY.glob(f"Quench/[qQ]*_C1_{sensor}_ai.txt")
    )
    if not matches:
        raise FileNotFoundError(f"no Quench file for sensor {sensor!r}")
    rows = []
    for line in matches[0].read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        time_value, sensor_value = line.split()
        rows.append((float(time_value), float(sensor_value)))
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def read_slow_signal(sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """Slow-control archive signal by full stem, e.g. 'TT2113C_ai'."""
    file_path = MEASUREMENT_DIRECTORY / "Slow" / f"{sensor}.txt"
    times, values = [], []
    for line in file_path.read_text(encoding="latin-1").splitlines():
        parts = line.strip().split()
        if len(parts) != 3 or parts[0].startswith(("#", "00/00")):
            continue
        if parts[2] == "#N/A":
            continue
        try:
            stamp = datetime.strptime(
                f"{parts[0]} {parts[1][:15]}", "%d/%m/%Y %H:%M:%S.%f"
            )
        except ValueError:
            continue
        times.append((stamp - QUENCH_TIMESTAMP).total_seconds())
        values.append(float(parts[2]))
    return np.array(times), np.array(values)


def merged_schedule(
    slow_sensor: str,
    event_sensor: str,
    scale: float = 1.0,
    window: tuple[float, float] = (-4.0e3, 9.0),
    slow_tail: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow archive up to the event window, fast file within it.

    Returns (measured time, value * scale) restricted to ``window``,
    strictly increasing in time. With ``slow_tail`` the slow archive
    also continues the trace past the end of the fast record, keeping
    one sample beyond ``window`` so a plotted line reaches the axis
    edge. (Off by default: the run controllers hold the last fast
    sample instead.)
    """
    slow_time, slow_value = read_slow_signal(slow_sensor)
    event_time, event_value = read_event_signal(event_sensor)
    keep_slow = (slow_time >= window[0]) & (slow_time < event_time[0])
    time = np.concatenate((slow_time[keep_slow], event_time))
    value = np.concatenate((slow_value[keep_slow], event_value))
    if slow_tail:
        after_event = slow_time > event_time[-1]
        time = np.concatenate((time, slow_time[after_event]))
        value = np.concatenate((value, slow_value[after_event]))
    in_window = time <= window[1]
    if slow_tail and not in_window.all():
        in_window[np.argmin(in_window)] = True
    return time[in_window], value[in_window] * scale


def discharge_current_template() -> tuple[np.ndarray, np.ndarray]:
    """Measured coil current from the decay onset, time-shifted to zero.

    The fast IAL-C1 record shows constant current until ~0.645 s, a
    switch phase to ~1.35 s and an exponential tail (tau = 1.96 s). The
    template starts at the last sample at which I >= 99.9 % of the
    pre-quench mean and is decimated to ~2000 points.
    """
    time, current = read_event_signal("IAL-C1")
    initial = current[time < 0.0].mean()
    onset_index = int(np.argmax(current < 0.999 * initial))
    template_time = time[onset_index:] - time[onset_index]
    template_current = current[onset_index:]
    stride = max(1, template_time.size // 2000)
    return template_time[::stride], template_current[::stride]
