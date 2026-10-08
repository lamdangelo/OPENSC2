"""Transitional re-export shim: utility_functions.checkpoint moved to adapters.output.checkpoint
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.output.checkpoint as _target

warnings.warn(
    "utility_functions.checkpoint has moved to adapters.output.checkpoint; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
