"""Transitional re-export shim: utility_functions.step_matrix_construction moved to numerics.step_matrix_construction
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import numerics.step_matrix_construction as _target

warnings.warn(
    "utility_functions.step_matrix_construction has moved to numerics.step_matrix_construction; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
