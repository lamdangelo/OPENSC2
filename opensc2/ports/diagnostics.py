"""Diagnostics sink port: where the numerics drop state dumps on failure.

The solvers used to write ``.npz`` files into the current working directory
when a solution turned non-finite. Writing files is not the numerics'
business: they now hand the arrays to a :class:`DiagnosticsSink`. The
default is :data:`NULL_DIAGNOSTICS` (no file); the Simulation facade wires
the npz writer of ``adapters.output.diagnostics_npz`` so production runs
keep producing the same dump files as before.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class DiagnosticsSink(Protocol):
    def dump(self, name: str, **arrays: np.ndarray) -> Optional[str]:
        """Store the named arrays under ``name``; return a human-readable
        location (file path) or None when nothing was stored."""
        ...


class NullDiagnosticsSink:
    """Discards every dump."""

    def dump(self, name: str, **arrays: np.ndarray) -> Optional[str]:
        return None


NULL_DIAGNOSTICS = NullDiagnosticsSink()


def dumped_to(location: Optional[str]) -> str:
    """Suffix for error messages: '' when nothing was dumped."""
    return f" State dumped to {location}." if location else ""
