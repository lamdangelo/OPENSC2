"""Transitional re-export shim: utility_functions.transient_solution_functions moved to numerics.transient_solution_functions
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import numerics.transient_solution_functions as _target

warnings.warn(
    "utility_functions.transient_solution_functions has moved to numerics.transient_solution_functions; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
