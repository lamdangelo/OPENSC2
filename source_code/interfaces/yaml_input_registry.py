"""Transitional re-export shim: interfaces.yaml_input_registry moved to adapters.input.yaml_input_registry
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.input.yaml_input_registry as _target

warnings.warn(
    "interfaces.yaml_input_registry has moved to adapters.input.yaml_input_registry; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
