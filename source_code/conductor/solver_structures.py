"""Typed structures of the coupled thermal-hydraulic solver.

These dataclasses replace the plain dictionaries that the solver used to
carry around (``dict_N_equation``, ``dict_band``, ``dict_Step`` and
``dict_norm``), giving every quantity a descriptive, attribute-checked name.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EquationCounts:
    """Numbers of equations of the coupled thermal-hydraulic system.

    Replaces ``conductor.dict_N_equation``.
    """
    fluid_equations: int   # was ["FluidComponent"]: 3 * number of fluid components
    strand_equations: int  # was ["StrandComponent"]
    jacket_equations: int  # was ["JacketComponent"]
    solid_equations: int   # was ["SolidComponent"]
    # was ["NODOFS"]: equations per mesh node (fluid + solid)
    degrees_of_freedom_per_node: int = 0
    # was ["NODOFS2"]: equations per element (two nodes)
    degrees_of_freedom_per_element: int = 0
    # was ["Total"]: equations of the whole system; needs the mesh, so it is
    # assigned at conductor initialization.
    total_equations: int = 0


@dataclass
class BandStructure:
    """Band structure of the assembled system matrix.

    Replaces ``conductor.dict_band``. The band storage keeps matrix row
    ``i`` in storage column ``i``: entry ``(i, j)`` lives at storage row
    ``number_of_subdiagonals + j - i``.
    """
    half_bandwidth: int          # was ["Half"]: subdiagonals + main diagonal
    number_of_subdiagonals: int  # was ["Main_diag"] (= number of superdiagonals)
    full_bandwidth: int          # was ["Full"]: total stored diagonals


@dataclass
class TimeIntegrationState:
    """Solution and load arrays of the transient time integration.

    Replaces ``conductor.dict_Step``. The trailing dimension holds the time
    levels required by the integration method (2 for Backward Euler /
    Crank-Nicolson, 4 for Adams-Moulton).
    """
    load_vector: np.ndarray                     # was ["SYSLOD"]
    solution: np.ndarray                        # was ["SYSVAR"]
    adams_moulton_matrices: np.ndarray = None   # was ["AM4_AA"]


@dataclass
class SolutionNorms:
    """Norms of the solution and of its change over the last time step.

    Replaces ``conductor.dict_norm``.
    """
    solution: np.ndarray  # was ["Solution"]
    change: np.ndarray    # was ["Change"]
