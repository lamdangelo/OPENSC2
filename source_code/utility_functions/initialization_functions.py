"""Transitional re-export shim: utility_functions.initialization_functions moved to conductor.initialization_functions
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import conductor.initialization_functions as _target

warnings.warn(
    "utility_functions.initialization_functions has moved to conductor.initialization_functions; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
