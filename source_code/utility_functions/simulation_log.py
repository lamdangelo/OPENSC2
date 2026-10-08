"""Transitional re-export shim: utility_functions.simulation_log moved to adapters.output.simulation_log
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.output.simulation_log as _target

warnings.warn(
    "utility_functions.simulation_log has moved to adapters.output.simulation_log; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
