"""
This module evaluates the full inductance matrix (self + mutual) for the
current-carrying strand components.  Two methods are supported:

* **Analytical** (``InductanceMode.ANALYTICAL``): exact Neumann integral
  solution for finite-length straight segments, with two alternative
  self-inductance formulae (``SelfInductanceMode.MODE_1`` and
  ``SelfInductanceMode.MODE_2``).

* **Approximate** (``InductanceMode.APPROXIMATED``): numerical double-
  integration of the Neumann kernel via ``scipy.integrate.dblquad``.

The public entry point is :func:`build_inductance_matrix`, which dispatches
to the appropriate internal path based on the conductor's operation flags.

Ported from the private methods of the Conductor class:
    __inductance_analytical_calculation
    __mutual_inductance
    __vertex_to_vertex_distance
    __self_inductance_mode1
    __self_inductance_mode2
    __inductance_approximate_calculation
    __mutual_inductance_approximate
    __self_inductance_approximate
"""

import numpy as np
from scipy import constants, integrate

from electromagnetics.electromagnetic_flags import InductanceMode, SelfInductanceMode


def build_inductance_matrix(conductor: object) -> None:
    """Compute and store the full inductance matrix in ``conductor.inductance_matrix``.

    Dispatches to the analytical or approximate calculation based on
    ``conductor.operations.inductance_mode``.

    Args:
        conductor: Conductor object whose nodal coordinates, connectivity
            matrix, inventory, and operation flags are already initialised.

    Raises:
        ValueError: if ``conductor.operations.inductance_mode`` is not a
            recognised ``InductanceMode`` value.
    """
    mode = conductor.operations.inductance_mode

    if mode == InductanceMode.ANALYTICAL:
        _analytical_inductance(conductor, conductor.operations.self_inductance_mode)
    elif mode == InductanceMode.APPROXIMATED:
        _approximate_inductance(conductor)
    else:
        raise ValueError(
            f"{conductor.identifier}: unrecognised inductance mode {mode!r}. "
            f"Expected one of {list(InductanceMode)}."
        )


# ---------------------------------------------------------------------------
# ANALYTICAL INDUCTANCE
# ---------------------------------------------------------------------------

def _analytical_inductance(
    conductor: object, self_mode: SelfInductanceMode
) -> None:
    """Evaluate the inductance matrix analytically using the Neumann formula.

    Computes:
    * Mutual inductances between all pairs of strand elements.
    * Self inductances using the formula selected by ``self_mode``.
    * Internal inductances (``L / 2`` per element).

    The result is scaled by ``mu_0 / (4 pi)`` and stored in
    ``conductor.inductance_matrix``.

    Args:
        conductor: Conductor object.
        self_mode: Formula variant for self-inductance evaluation.

    Raises:
        ValueError: if ``self_mode`` is not ``MODE_1`` or ``MODE_2``.
    """
    if self_mode not in (SelfInductanceMode.MODE_1, SelfInductanceMode.MODE_2):
        raise ValueError(
            f"{conductor.identifier}: self_inductance_mode must be "
            f"{SelfInductanceMode.MODE_1} or {SelfInductanceMode.MODE_2}; "
            f"got {self_mode!r}."
        )

    ABSTOL = 1e-6

    # Element lengths for all strand elements
    lmod = (
        (
            (
                conductor.nodal_coordinates.iloc[
                    conductor.connectivity_matrix.loc["StrandComponent", "end"], :
                ]
                - conductor.nodal_coordinates.iloc[
                    conductor.connectivity_matrix.loc["StrandComponent", "start"], :
                ]
            )
            ** 2
        )
        .sum(axis=1)
        .apply(np.sqrt)
    )

    mutual_inductance = np.zeros(conductor.inductance_matrix.shape)

    if conductor.inventory.strands.number > 1:
        for ii in range(conductor.total_elements_current_carriers - 1):
            mutual_inductance = _mutual_inductance_analytical(
                conductor, lmod, ii, mutual_inductance, ABSTOL
            )

    self_inductance_fn = {
        SelfInductanceMode.MODE_1: _self_inductance_mode1,
        SelfInductanceMode.MODE_2: _self_inductance_mode2,
    }
    self_inductance = self_inductance_fn[self_mode](conductor, lmod)

    internal_inductance = lmod / 2.0

    conductor.inductance_matrix = (
        constants.mu_0
        / (4.0 * constants.pi)
        * (
            np.diag(self_inductance + internal_inductance)
            + mutual_inductance
            + mutual_inductance.T
        )
    )


def _mutual_inductance_analytical(
    conductor: object,
    lmod: object,
    ii: int,
    matrix: np.ndarray,
    abstol: float = 1e-6,
) -> np.ndarray:
    """Evaluate mutual inductances for element ``ii`` against all later elements.

    Uses the analytical Neumann formula for finite straight segments.

    Args:
        conductor: Conductor object.
        lmod: Series of element lengths for all strand elements.
        ii: Index of the source element.
        matrix: Accumulator matrix of shape ``(n_elements, n_elements)``; is
            updated in-place and returned.
        abstol: Absolute tolerance to avoid division by zero for co-planar
            segments.

    Returns:
        Updated ``matrix`` with entries ``matrix[ii, jj]`` filled for all
        ``jj > ii``.
    """
    jj = np.r_[ii + 1 : conductor.total_elements_current_carriers]
    ll = lmod[jj]
    mm = lmod[ii]
    len_jj = conductor.total_elements_current_carriers - (ii + 1)

    rr = {
        key: _vertex_to_vertex_distance(conductor, key, ii, jj)
        for key in ("end_end", "end_start", "start_start", "start_end")
    }

    alpha2 = (
        rr["start_end"] ** 2
        - rr["start_start"] ** 2
        + rr["end_start"] ** 2
        - rr["end_end"] ** 2
    )

    cos_eps = np.minimum(np.maximum(alpha2 / (2 * ll * mm), -1.0), 1.0)
    sin_eps = np.sin(np.arccos(cos_eps))

    dd = 4 * ll ** 2 * mm ** 2 - alpha2 ** 2
    mu = (
        ll
        * (
            2 * mm ** 2 * (rr["end_start"] ** 2 - rr["start_start"] ** 2 - ll ** 2)
            + alpha2 * (rr["start_end"] ** 2 - rr["start_start"] ** 2 - mm ** 2)
        )
        / dd
    )
    nu = (
        mm
        * (
            2 * ll ** 2 * (rr["start_end"] ** 2 - rr["start_start"] ** 2 - mm ** 2)
            + alpha2 * (rr["end_start"] ** 2 - rr["start_start"] ** 2 - ll ** 2)
        )
        / dd
    )
    d2 = rr["start_start"] ** 2 - mu ** 2 - nu ** 2 + 2 * mu * nu * cos_eps

    # Avoid rounding errors for co-planar segments
    d2[d2 < abstol ** 2] = 0
    d0 = np.sqrt(d2)

    omega = (
        np.arctan(
            (d2 * cos_eps + (mu + ll) * (nu + mm) * sin_eps ** 2)
            / (d0 * rr["end_end"] * sin_eps)
        )
        - np.arctan(
            (d2 * cos_eps + (mu + ll) * nu * sin_eps ** 2)
            / (d0 * rr["end_start"] * sin_eps)
        )
        + np.arctan(
            (d2 * cos_eps + mu * nu * sin_eps ** 2)
            / (d0 * rr["start_start"] * sin_eps)
        )
        - np.arctan(
            (d2 * cos_eps + mu * (nu + mm) * sin_eps ** 2)
            / (d0 * rr["start_end"] * sin_eps)
        )
    )
    omega[d0 == 0.0] = 0.0

    pp = np.zeros((len_jj, 5), dtype=float)
    pp[:, 0] = (ll + mu) * np.arctanh(mm / (rr["end_end"] + rr["end_start"]))
    pp[:, 1] = -nu * np.arctanh(ll / (rr["end_start"] + rr["start_start"]))
    pp[:, 2] = (mm + nu) * np.arctanh(ll / (rr["end_end"] + rr["start_end"]))
    pp[:, 3] = -mu * np.arctanh(mm / (rr["start_start"] + rr["start_end"]))
    pp[:, 4] = d0 * omega / sin_eps

    # Filter degenerate cases (e.g. consecutive collinear segments)
    pp[np.isnan(pp)] = 0.0
    pp[np.isinf(pp)] = 0.0

    matrix[ii, jj] = (
        2 * cos_eps * (pp[:, 0] + pp[:, 1] + pp[:, 2] + pp[:, 3])
        - cos_eps * pp[:, 4]
    )
    return matrix


def _vertex_to_vertex_distance(
    conductor: object, key: str, ii: int, jj: np.ndarray
) -> np.ndarray:
    """Compute Euclidean distances between element vertices.

    Args:
        conductor: Conductor object with nodal coordinates and strand
            connectivity matrix.
        key: One of ``"start_end"``, ``"start_start"``, ``"end_end"``,
            ``"end_start"``, indicating which vertices of elements ``ii``
            and ``jj`` to compare.
        ii: Source element index.
        jj: Array of target element indices.

    Returns:
        Array of distances, one per element in ``jj``.
    """
    cols = key.split("_")
    cm = conductor.connectivity_matrix.loc["StrandComponent"]
    return (
        (
            (
                conductor.nodal_coordinates.iloc[
                    cm.iloc[jj, cm.columns.get_loc(cols[0])], :
                ]
                - conductor.nodal_coordinates.iloc[
                    cm.iat[ii, cm.columns.get_loc(cols[1])], :
                ]
            )
            ** 2
        )
        .sum(axis="columns")
        .apply(np.sqrt)
    )


def _self_inductance_mode1(conductor: object, lmod: object) -> np.ndarray:
    """Compute self-inductances using mode-1 formula (arcsinh-based).

    Args:
        conductor: Conductor object with strand inventory.
        lmod: Series of element lengths for all strand elements.

    Returns:
        Array of self-inductance values, one per strand element.
    """
    self_inductance = np.zeros(lmod.shape)
    n_strands = conductor.inventory.strands.number

    for ii, strand in enumerate(conductor.inventory.strands.collection):
        l_ii = lmod[ii::n_strands]
        self_inductance[ii::n_strands] = 2 * l_ii * (
            np.arcsinh(l_ii / strand.radius)
            - np.sqrt(1.0 + (strand.radius / l_ii) ** 2)
            + strand.radius / l_ii
        )
    return self_inductance


def _self_inductance_mode2(conductor: object, lmod: object) -> np.ndarray:
    """Compute self-inductances using mode-2 formula (logarithm-based).

    Args:
        conductor: Conductor object with strand inventory.
        lmod: Series of element lengths for all strand elements.

    Returns:
        Array of self-inductance values, one per strand element.
    """
    self_inductance = np.zeros(lmod.shape)
    n_strands = conductor.inventory.strands.number

    for ii, strand in enumerate(conductor.inventory.strands.collection):
        l_ii = lmod[ii::n_strands]
        self_inductance[ii::n_strands] = 2 * (
            l_ii
            * np.log(
                (l_ii + np.sqrt(l_ii ** 2 + strand.radius ** 2)) / strand.radius
            )
            - np.sqrt(l_ii ** 2 + strand.radius ** 2)
            + l_ii / 4
            + strand.radius
        )
    return self_inductance


# ---------------------------------------------------------------------------
# APPROXIMATE INDUCTANCE
# ---------------------------------------------------------------------------

def _approximate_inductance(conductor: object) -> None:
    """Evaluate the inductance matrix via numerical double integration.

    Uses ``scipy.integrate.dblquad`` to approximate the Neumann kernel
    integral for each element pair. Intended as a fallback when the
    analytical formula is not applicable.

    The result is scaled by ``mu_0 / (4 pi)`` and stored in
    ``conductor.inductance_matrix``.

    Args:
        conductor: Conductor object.
    """
    ll = (
        conductor.nodal_coordinates.iloc[
            conductor.connectivity_matrix.loc["StrandComponent", "end"], :
        ]
        - conductor.nodal_coordinates.iloc[
            conductor.connectivity_matrix.loc["StrandComponent", "start"], :
        ]
    )
    lmod = (ll ** 2).sum(axis=1).apply(np.sqrt)

    mutual_inductance = np.zeros(conductor.inductance_matrix.shape)
    if conductor.inventory.strands.number > 1:
        mutual_inductance = _mutual_inductance_approximate(
            conductor, ll, mutual_inductance
        )

    self_inductance = _self_inductance_approximate(conductor, lmod)
    internal_inductance = lmod.to_numpy() / 2.0

    conductor.inductance_matrix = (
        constants.mu_0
        / (4.0 * constants.pi)
        * (
            np.diag(self_inductance + internal_inductance)
            + mutual_inductance
            + mutual_inductance.T
        )
    )


def _mutual_inductance_approximate(
    conductor: object, ll: object, matrix: np.ndarray
) -> np.ndarray:
    """Approximate mutual inductances via numerical double integration.

    Each off-diagonal entry ``matrix[ii, jj]`` is computed by integrating
    the dot product of the direction vectors scaled by the inverse distance
    between points on the two segments.

    Args:
        conductor: Conductor object.
        ll: DataFrame of element direction vectors (end minus start).
        matrix: Zero-initialised accumulator of shape
            ``(n_elements, n_elements)``; updated in-place and returned.

    Returns:
        Updated ``matrix``.
    """
    ABSTOL = 1e-4
    RELTOL = 1e-4

    def _reverse_distance(xi, xj, qi, vi, qj, vj):
        return 1.0 / np.sqrt(
            (qi[0] + xi * vi[0] - qj[0] - xj * vj[0]) ** 2
            + (qi[1] + xi * vi[1] - qj[1] - xj * vj[1]) ** 2
            + (qi[2] + xi * vi[2] - qj[2] - xj * vj[2]) ** 2
        )

    cm_start = conductor.connectivity_matrix.loc["StrandComponent", "start"]

    for ii in range(conductor.total_elements_current_carriers):
        qi = conductor.nodal_coordinates.iloc[cm_start, :].iloc[ii, :].to_numpy()
        vi = ll.iloc[ii, :].to_numpy()
        for jj in range(ii + 1, conductor.total_elements_current_carriers):
            qj = conductor.nodal_coordinates.iloc[cm_start, :].iloc[jj, :].to_numpy()
            vj = ll.iloc[jj, :].to_numpy()
            matrix[ii, jj] = np.sum(vi * vj) * integrate.dblquad(
                _reverse_distance,
                0.0,
                1.0,
                0.0,
                1.0,
                (qi, vi, qj, vj),
                ABSTOL,
                RELTOL,
            )[0]

    return matrix


def _self_inductance_approximate(conductor: object, lmod: object) -> np.ndarray:
    """Approximate self-inductances using the logarithm-based mode-2 formula.

    The approximate and analytical mode-2 formulae are mathematically
    identical; this function exists to keep the approximate path self-contained.

    Args:
        conductor: Conductor object with strand inventory.
        lmod: Series of element lengths for all strand elements.

    Returns:
        Array of self-inductance values, one per strand element.
    """
    self_inductance = np.zeros(lmod.shape)
    n_strands = conductor.inventory.strands.number

    for ii, strand in enumerate(conductor.inventory.strands.collection):
        l_ii = lmod[ii::n_strands]
        self_inductance[ii::n_strands] = 2 * (
            l_ii
            * np.log(
                (l_ii + np.sqrt(l_ii ** 2 + strand.radius ** 2)) / strand.radius
            )
            - np.sqrt(l_ii ** 2 + strand.radius ** 2)
            + l_ii / 4
            + strand.radius
        )

    return self_inductance
