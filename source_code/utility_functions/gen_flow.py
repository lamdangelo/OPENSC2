"""Transitional re-export shim: utility_functions.gen_flow moved to numerics.initial_flow
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import numerics.initial_flow as _target

warnings.warn(
    "utility_functions.gen_flow has moved to numerics.initial_flow; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
