"""
This module contains the CouplingMatrix class to store the coupling data between conductor components.
"""

import numpy as np
import pandas as pd
from typing import Generic, TypeVar, Union

T = TypeVar("T")

ComponentIndex = Union[int, str]


class CouplingMatrix(Generic[T]):
    """Symmetric component-pair matrix built from one sheet of the conductor
    coupling workbook.

    Only the upper triangle is stored; index pairs are folded so that
    ``matrix_object[a, b]`` and ``matrix_object[b, a]`` address the same entry.
    Components can be addressed either by integer position or by identifier
    string (the row/column labels of the workbook sheet, e.g. "CHAN_1" or
    "Environment").
    """

    def __init__(self, data_frame: pd.DataFrame, dtype: type[T]):
        self.matrix = np.triu(data_frame.to_numpy(dtype=dtype))  # np.ndarray
        self.component_names = data_frame.columns.tolist()  # List[str]
        self.type = dtype
        self._name_to_index = {
            name: index for index, name in enumerate(self.component_names)
        }

    def _resolve_index(self, index: tuple[ComponentIndex, ComponentIndex]) -> tuple[int, int]:
        """Convert component identifiers to integer positions and fold the
        pair onto the stored upper triangle."""
        i, j = index
        if isinstance(i, str):
            i = self._name_to_index[i]
        if isinstance(j, str):
            j = self._name_to_index[j]
        if i <= j:
            return i, j
        return j, i

    def __getitem__(self, index: tuple[ComponentIndex, ComponentIndex]) -> T:
        i, j = self._resolve_index(index)
        return self.matrix[i, j]

    def __setitem__(self, index: tuple[ComponentIndex, ComponentIndex], value: T) -> None:
        i, j = self._resolve_index(index)
        self.matrix[i, j] = value

    def check_for_negative_data(self) -> bool:
        """Checks whether the data contains any negative value."""
        return np.any(self.matrix < 0)
