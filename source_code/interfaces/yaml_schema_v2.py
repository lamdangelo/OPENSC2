"""Transitional re-export shim: interfaces.yaml_schema_v2 moved to adapters.input.yaml_schema_v2
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.input.yaml_schema_v2 as _target

warnings.warn(
    "interfaces.yaml_schema_v2 has moved to adapters.input.yaml_schema_v2; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
