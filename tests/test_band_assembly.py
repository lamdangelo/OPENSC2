"""Bit-identity tests of the banded system-matrix assembly kernels against
the original per-(local row, local column) scatter loop."""

from types import SimpleNamespace

import numpy as np
import pytest

from utility_functions import step_matrix_construction as smc


def reference_scatter(fmat, batch, number_of_elements, nodofs, half):
    """The original implementation (kept here as the reference)."""
    first_columns = nodofs * np.arange(number_of_elements)
    for local_row in range(half):
        columns = first_columns + local_row
        for local_col in range(half):
            fmat[half - 1 - local_row + local_col, columns] += batch[:, local_row, local_col]
    return fmat


def make_case(number_of_elements, nodofs, seed):
    half = 2 * nodofs
    full = 4 * nodofs - 1
    total = nodofs * (number_of_elements + 1)
    rng = np.random.default_rng(seed)
    batch = rng.standard_normal((number_of_elements, half, half))
    return half, full, total, batch


KERNELS = [smc._scatter_into_band_numpy]
if hasattr(smc, "_scatter_into_band_numba"):
    KERNELS.append(smc._scatter_into_band_numba)


@pytest.mark.parametrize("kernel", KERNELS, ids=lambda k: k.__name__)
@pytest.mark.parametrize("number_of_elements, nodofs", [(1, 1), (2, 1), (3, 2), (7, 3), (40, 5), (13, 31)])
def test_kernels_match_reference_bit_for_bit(kernel, number_of_elements, nodofs):
    half, full, total, batch = make_case(number_of_elements, nodofs, seed=number_of_elements)
    expected = reference_scatter(np.zeros((full, total)), batch, number_of_elements, nodofs, half)
    result = np.zeros((full, total))
    kernel(result, batch, number_of_elements, nodofs, half)
    np.testing.assert_array_equal(result, expected)


def test_assemble_system_matrices_uses_kernel_on_all_four_matrices():
    number_of_elements, nodofs = 9, 4
    half, full, total, _ = make_case(number_of_elements, nodofs, seed=0)
    rng = np.random.default_rng(1)
    batches = [rng.standard_normal((number_of_elements, half, half)) for _ in range(4)]
    element_matrices = smc.ElementMatrices(*batches)
    system_matrices = smc.SystemMatrices(*(np.zeros((full, total)) for _ in range(4)))
    conductor = SimpleNamespace(
        band=SimpleNamespace(half_bandwidth=half),
        equation_counts=SimpleNamespace(degrees_of_freedom_per_node=nodofs),
        mesh=SimpleNamespace(number_of_elements=number_of_elements),
    )
    smc.assemble_system_matrices(system_matrices, element_matrices, conductor)
    for matrix, batch in zip(system_matrices, batches):
        expected = reference_scatter(np.zeros((full, total)), batch, number_of_elements, nodofs, half)
        np.testing.assert_array_equal(matrix, expected)


def test_numba_kernel_is_selected_when_available():
    pytest.importorskip("numba")
    assert smc._scatter_element_matrices_into_band is smc._scatter_into_band_numba
