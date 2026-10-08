"""Transitional re-export shim: conductor.input_loader moved to
adapters.input.conductor_loader (Stage 4a). Removed in Stage 9."""

import sys
import warnings

import adapters.input.conductor_loader as _target

warnings.warn(
    "conductor.input_loader has moved to adapters.input.conductor_loader",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _target
