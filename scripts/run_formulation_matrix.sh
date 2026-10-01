#!/usr/bin/env bash
# Full run matrix of the formulation verification suite, cheapest tiers
# first, the two formulations as two parallel chains (each solver run is
# single-threaded). Finished runs (manifest present) are skipped, so the
# script is restartable. Logs in <results>/logs/.
#
#   scripts/run_formulation_matrix.sh [results_dir]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${1:-$ROOT/results}"
PYTHON="$ROOT/.venv/bin/python"
DRIVER="$ROOT/tests/verification/formulation_suite/driver.py"
mkdir -p "$RESULTS/logs"

chain() {
  local formulation="$1"
  local log="$RESULTS/logs/matrix_${formulation}.log"
  {
    echo "=== chain $formulation started $(date) ==="
    # Cheap tiers at all levels.
    "$PYTHON" "$DRIVER" --tier 1 --formulation "$formulation" --level h,h2,h4 --results-dir "$RESULTS"
    "$PYTHON" "$DRIVER" --tier 4 --formulation "$formulation" --level h,h2,h4 --results-dir "$RESULTS"
    "$PYTHON" "$DRIVER" --tier 3a --formulation "$formulation" --level h,h2,h4 --results-dir "$RESULTS"
    "$PYTHON" "$DRIVER" --tier 3b --formulation "$formulation" --level h,h2,h4 --results-dir "$RESULTS"
    # Tier 2: proportional series h, h2 and the dx / dt series, then h4 last.
    "$PYTHON" "$DRIVER" --tier 2 --formulation "$formulation" --level h,h2 --results-dir "$RESULTS"
    "$PYTHON" "$DRIVER" --tier 2 --formulation "$formulation" --level h4 --results-dir "$RESULTS"
    echo "=== chain $formulation finished $(date) ==="
  } >> "$log" 2>&1
}

chain velocity &
chain mass_flow &
wait
echo "matrix finished $(date)" >> "$RESULTS/logs/matrix.log"
