"""Transitional re-export shim: utility_functions.output moved to adapters.output.solution_writer
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.output.solution_writer as _target

warnings.warn(
    "utility_functions.output has moved to adapters.output.solution_writer; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
