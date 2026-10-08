"""Transitional re-export shim: utility_functions.plots moved to adapters.plotting.plots
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.plotting.plots as _target

warnings.warn(
    "utility_functions.plots has moved to adapters.plotting.plots; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
