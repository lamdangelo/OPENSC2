"""
This module defines the external current for the W7-X conductor and writes
it down as an Excel input file for OPENSC².

Author: Laura D'Angelo, Konrad Risse
"""

import numpy as np
import pandas as pd


def external_current(t, tau_detection):
    """
    External current of the W7-X.

    Parameters
    ----------
    t : float or np.ndarray
        Time [s]
    tau_detection : float
        Delay before current evolution starts [s]

    Returns
    -------
    float or np.ndarray
        Current [A]
    """

    t1 = t - tau_detection
    t2 = tau_detection + 3.72
    t3 = tau_detection + 30.0

    I0 = 15320.0

    # Phase masks (works for scalar or array input)
    t = np.asarray(t)

    I = np.zeros_like(t, dtype=float)

    # Phase 1: constant current
    mask1 = t <= tau_detection
    I[mask1] = I0

    # Phase 2: linear decay
    mask2 = (t > tau_detection) & (t < t2)
    I[mask2] = I0 - 2128.9 * (t[mask2] - tau_detection)

    # Phase 3: exponential decay
    mask3 = (t >= t2) & (t <= t3)
    I[mask3] = 17428.0 * np.exp(-0.233 * (t[mask3] - tau_detection))

    # Phase 4: after event → current goes to 0
    mask4 = t > t3
    I[mask4] = 0.0

    return I if I.shape != () else float(I)


def write_current_to_excel_time_only(
    times,
    current_values,
    file_path,
    sheet_name,
    space_value=0.0
):
    """
    Write a time-dependent current profile in the format required by
    build_interpolator() for the 'time_only' case.

    Parameters
    ----------
    times : array-like
        Time values [s]
    current_values : array-like
        Current values [A], same length as times
    file_path : str
        Output Excel file path (e.g. external_current.xlsx)
    sheet_name : str
        Excel sheet name (must match self.identifier in your code)
    space_value : float
        Single spatial coordinate (required by loader but not used)
    """

    times = np.asarray(times, dtype=float)
    current_values = np.asarray(current_values, dtype=float)

    if times.shape != current_values.shape:
        raise ValueError("times and current_values must have the same shape")

    # Build dataframe in REQUIRED structure:
    #
    # Row 0: time points (starting from column 1)
    # Row 1: space coordinate + values row
    data = np.zeros((2, len(times) + 1), dtype=float)

    # First row: time headers
    data[0, 1:] = times

    # First column: space coordinate
    data[1, 0] = space_value

    # Second row: current values
    data[1, 1:] = current_values

    df = pd.DataFrame(data)

    # Write to Excel
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)

    print(f"Saved time-only current profile to {file_path} (sheet: {sheet_name})")


def main():
    times = np.linspace(0, 25, 100)
    tau_detection = 0.895
    current_values = external_current(times, tau_detection)
    file_path = "./W7-X/external_current.xlsx"
    sheet_name = "STR_MIX_1"
    write_current_to_excel_time_only(times, current_values, file_path, sheet_name)


if __name__=="__main__":
    main()