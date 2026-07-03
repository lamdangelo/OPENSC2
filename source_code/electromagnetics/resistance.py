"""
This module builds the electric resistance matrix for the current-carrying
strand components (StrandMixedComponent, StrandStabilizerComponent,
StackComponent).

The resistance matrix is a sparse diagonal matrix of shape
``(n_elements, n_elements)`` whose entries are the per-element electric
resistances ``R = rho * L / A``.  Resistivity and cross-section are
evaluated by each strand object via its ``get_electric_resistance`` method,
which accounts for material properties, operating temperature, and magnetic
field.

Ported from the private method of the Conductor class:
    __build_electric_resistance_matrix
"""

import numpy as np
from scipy.sparse import diags


def build_resistance_matrix(conductor: object) -> None:
    """Build the diagonal electric resistance matrix for current carriers.

    Iterates over all strand objects in the inventory, calling each strand's
    ``get_electric_resistance`` method to obtain element resistances.  The
    results are assembled into a sparse CSR diagonal matrix and stored in
    ``conductor.electric_resistance_matrix``.

    Args:
        conductor: Conductor object with ``inventory.strands`` and
            ``total_elements_current_carriers`` set up, and with operating
            conditions already evaluated (so that resistivity is available).
    """
    n_elements = conductor.total_elements_current_carriers
    n_strands = conductor.inventory.strands.number

    resistance = np.zeros(n_elements)
    for ii, strand in enumerate(conductor.inventory.strands.collection):
        resistance[ii::n_strands] = strand.get_electric_resistance(conductor)

    conductor.electric_resistance_matrix = diags(
        resistance,
        offsets=0,
        shape=(n_elements, n_elements),
        format="csr",
        dtype=float,
    )
