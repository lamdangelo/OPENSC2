"""Small geometric kernels shared by the mesh and the electric topology."""

import numpy as np


def row_norms(vectors: np.ndarray) -> np.ndarray:
    """Euclidean norm of every row of an ``(n, 3)`` array.

    The squares are added in column order, ``(dx**2 + dy**2) + dz**2``: the
    order the former pandas ``DataFrame.sum(axis=1)`` used, so the result is
    bit-identical to the table-based implementation it replaced.
    """
    squares = vectors ** 2
    return np.sqrt(squares[:, 0] + squares[:, 1] + squares[:, 2])
