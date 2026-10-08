"""Bitwise output-tree comparison for refactoring stages.

The regression tests compare against reference data within a 1e-5 column
scaled tolerance. Pure refactoring (moves, signature changes, dependency
inversion) must do better: the Output tree of a run must be byte-identical
before and after the change. This tool records a baseline of SHA-256 hashes
of every file in the Output tree of a TDD example and checks a later run
against it.

Usage (from the repository root, with the project virtualenv)::

    python tests/tools/compare_trees.py record CASE_1 [--baseline-dir DIR]
    python tests/tools/compare_trees.py check  CASE_1 [--baseline-dir DIR]
    python tests/tools/compare_trees.py record all

``CASE_2`` and ``CASE_3`` run the shortened CI scenarios of the respective
regression tests. Baselines default to ``~/.cache/opensc2_tree_baselines``
because the hashes are only meaningful on the machine (BLAS, CPU) that
recorded them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIRECTORY = REPOSITORY_ROOT / "tests"
SOURCE_DIRECTORY = REPOSITORY_ROOT / "source_code"
for directory in (SOURCE_DIRECTORY, TESTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

os.environ.setdefault("MPLBACKEND", "Agg")

DEFAULT_BASELINE_DIRECTORY = Path.home() / ".cache" / "opensc2_tree_baselines"

CASES = {
    "CASE_1": ("CASE_1_ITER_like_LTS", None),
    "CASE_2": ("CASE_2_ENEA_HTS_CICC", "test_case2_regression"),
    "CASE_3": ("CASE_3_HTS_HVDC", "test_case3_regression"),
}


def output_tree_hashes(run_directory: Path) -> dict[str, str]:
    """{relative path: sha256} of every file in the single Output tree."""
    output_roots = list(run_directory.glob("*/*/Output"))
    if len(output_roots) != 1:
        raise RuntimeError(f"expected one Output tree, found {output_roots}")
    return {
        str(path.relative_to(output_roots[0])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output_roots[0].rglob("*"))
        if path.is_file()
    }


def run_case(case: str, input_format: str = "yaml") -> dict[str, str]:
    """Run one TDD example headless in a temporary directory; return hashes."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from regression_utilities import copy_input_files
    from simulation import Simulation

    example_name, shortening_module = CASES[case]
    input_directory = REPOSITORY_ROOT / "TDD_examples" / example_name
    with tempfile.TemporaryDirectory(prefix=f"opensc2_{case}_") as scratch:
        run_directory = Path(scratch) / example_name
        run_directory.mkdir()
        copy_input_files(input_directory, run_directory, input_format)
        if shortening_module is not None:
            module = __import__(shortening_module)
            module.shorten_scenario(run_directory)
        Simulation(str(run_directory)).run()
        return output_tree_hashes(run_directory)


def baseline_path(baseline_directory: Path, case: str, input_format: str) -> Path:
    return baseline_directory / f"{case}_{input_format}.json"


def record(case: str, baseline_directory: Path, input_format: str) -> None:
    hashes = run_case(case, input_format)
    baseline_directory.mkdir(parents=True, exist_ok=True)
    path = baseline_path(baseline_directory, case, input_format)
    path.write_text(json.dumps(hashes, indent=1, sort_keys=True))
    print(f"{case} [{input_format}]: recorded {len(hashes)} files -> {path}")


def check(case: str, baseline_directory: Path, input_format: str) -> bool:
    path = baseline_path(baseline_directory, case, input_format)
    if not path.exists():
        print(f"{case} [{input_format}]: no baseline at {path}")
        return False
    baseline = json.loads(path.read_text())
    hashes = run_case(case, input_format)
    missing = sorted(set(baseline) - set(hashes))
    extra = sorted(set(hashes) - set(baseline))
    changed = sorted(k for k in baseline if k in hashes and hashes[k] != baseline[k])
    if not (missing or extra or changed):
        print(f"{case} [{input_format}]: IDENTICAL ({len(hashes)} files)")
        return True
    print(f"{case} [{input_format}]: DIFFERENT")
    for label, names in (("missing", missing), ("extra", extra), ("changed", changed)):
        for name in names:
            print(f"  {label}: {name}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("record", "check"))
    parser.add_argument("case", choices=(*CASES, "all"))
    parser.add_argument("--format", default="yaml", choices=("yaml", "excel"))
    parser.add_argument(
        "--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIRECTORY
    )
    arguments = parser.parse_args(argv)
    cases = list(CASES) if arguments.case == "all" else [arguments.case]
    ok = True
    for case in cases:
        if arguments.action == "record":
            record(case, arguments.baseline_dir, arguments.format)
        else:
            ok &= check(case, arguments.baseline_dir, arguments.format)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
