"""npz implementation of the diagnostics sink port."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class NpzDiagnosticsSink:
    """Writes every dump as ``<directory>/<name>.npz`` (directory defaults to
    the current working directory at the time of the dump, which is what the
    solvers did before the port existed)."""

    def __init__(self, directory: Optional[str] = None):
        self.directory = directory

    def dump(self, name: str, **arrays: np.ndarray) -> Optional[str]:
        directory = self.directory if self.directory is not None else os.getcwd()
        path = os.path.join(directory, f"{name}.npz")
        np.savez(path, **arrays)
        return path
