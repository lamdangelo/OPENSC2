"""Transitional re-export shim: interfaces.coolprop_interface moved to adapters.fluid_properties.coolprop_provider
(docs/architecture_refactoring_plan.md, Stage 2). Imports resolve to the
very same module object; this shim is removed in Stage 9."""

import sys
import warnings

import adapters.fluid_properties.coolprop_provider as _target

warnings.warn(
    "interfaces.coolprop_interface has moved to adapters.fluid_properties.coolprop_provider; update the import",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
