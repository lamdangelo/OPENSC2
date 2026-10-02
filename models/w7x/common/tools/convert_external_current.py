"""One-off: convert a legacy ``external_current.xlsx`` auxiliary file into the
CSV auxiliary format OPENSC2 reads (``utility_functions.auxiliary_functions.
load_auxiliary_files``): one block per component, first column the component
identifier, then the legacy sheet rows (row 0 = flag + times, row 1 = flag +
values for a time-only table).

Usage:
    ../../../../.venv/bin/python convert_external_current.py \
        --xlsx <path>/external_current.xlsx --component STR_MIX_1 \
        --target <model>/external_current.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "source_code"))

from utility_functions.auxiliary_functions import load_auxiliary_files  # noqa: E402


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if float(value) == int(float(value)):
        return str(int(float(value)))
    return f"{float(value):.17g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--target", type=Path, required=True)
    arguments = parser.parse_args()

    frame, flag = load_auxiliary_files(arguments.xlsx, arguments.component)
    lines = [
        ",".join([arguments.component] + [format_value(v) for v in frame.iloc[row]])
        for row in range(frame.shape[0])
    ]
    arguments.target.write_text("\n".join(lines) + "\n")

    check, check_flag = load_auxiliary_files(arguments.target, arguments.component)
    pd.testing.assert_frame_equal(
        check.astype(float), frame.astype(float), check_dtype=False
    )
    assert float(check_flag) == float(flag), (check_flag, flag)
    print(
        f"wrote {arguments.target} ({frame.shape[0]} rows x {frame.shape[1]} "
        f"columns, flag {flag}); reload identical"
    )


if __name__ == "__main__":
    main()
