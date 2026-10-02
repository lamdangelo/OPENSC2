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

from electromagnetics.electromagnetic_flags import ElectricConductanceMode


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
        resistance[ii::n_strands] = combine_parallel_jacket_resistance(
            conductor, strand, strand.get_electric_resistance(conductor)
        )

    conductor.electric_resistance_matrix = diags(
        resistance,
        offsets=0,
        shape=(n_elements, n_elements),
        format="csr",
        dtype=float,
    )


def build_differential_resistance_matrix(conductor: object) -> None:
    """Build the diagonal differential-resistance matrix d(V_e)/d(I_e).

    Companion of :func:`build_resistance_matrix` for the Newton branch of
    the steady-state consistency solver: same element interleaving, but the
    entries are the exact derivatives of the per-element voltage with
    respect to the element current, evaluated at the operating point of the
    LAST ``build_resistance_matrix`` call (it consumes the regime data each
    strand stored there). Result in
    ``conductor.electric_differential_resistance_matrix``.

    Args:
        conductor: Conductor object on which ``build_resistance_matrix``
            (via ``electric_preprocessing``) has just run.
    """
    n_elements = conductor.total_elements_current_carriers
    n_strands = conductor.inventory.strands.number

    differential = np.zeros(n_elements)
    for ii, strand in enumerate(conductor.inventory.strands.collection):
        differential[ii::n_strands] = (
            combine_parallel_jacket_differential_resistance(
                conductor,
                strand,
                strand.get_electric_resistance_derivative(conductor),
            )
        )

    conductor.electric_differential_resistance_matrix = diags(
        differential,
        offsets=0,
        shape=(n_elements, n_elements),
        format="csr",
        dtype=float,
    )


def combine_parallel_jacket_differential_resistance(
    conductor: object, strand: object, strand_differential: np.ndarray
) -> np.ndarray:
    """Differential resistance of the strand-jacket parallel chain.

    Mirrors :func:`combine_parallel_jacket_resistance` (same jackets, same
    chaining order) but is side-effect free and propagates the derivative of
    the residual as assembled: with V = R_par(I) * I and
    R_par = R_eff(I) || R_j (R_j current-independent),

        d(R_par * I)/dI = R_par + (R_j / (R_eff + R_j))**2 * (d_eff - R_eff)

    where d_eff = d(R_eff * I)/dI of the chain so far. NOTE: this is NOT
    the parallel of d_eff with R_j — using that form silently degrades
    Newton convergence on quench fronts.

    Args:
        conductor: Conductor object (coupling matrices and mesh available).
        strand: the current-carrying strand component.
        strand_differential: per-element d(V)/d(I) of the strand alone (Ohm).

    Returns:
        np.ndarray: per-element differential resistance of the parallel
        chain in Ohm.
    """
    effective_resistance = strand._electric_resistance_strand_only
    effective_differential = strand_differential

    mode_matrix = conductor.coupling.electric_conductance_mode
    for jacket in conductor.inventory.jackets.collection:
        mode = ElectricConductanceMode.get_electric_conductance_mode(
            int(mode_matrix[strand.identifier, jacket.identifier])
        )
        if mode is ElectricConductanceMode.NO_CONDUCTANCE:
            continue
        jacket_resistance = (
            jacket.jacket_electrical_resistivity(jacket.gauss_fields)
            * conductor.mesh.element_lengths
            / (jacket.inputs.cross_section * jacket.inputs.cos_theta)
        )
        parallel_resistance = (
            effective_resistance * jacket_resistance
            / (effective_resistance + jacket_resistance)
        )
        effective_differential = parallel_resistance + (
            jacket_resistance / (effective_resistance + jacket_resistance)
        ) ** 2 * (effective_differential - effective_resistance)
        effective_resistance = parallel_resistance

    return effective_differential


def combine_parallel_jacket_resistance(
    conductor: object, strand: object, strand_resistance: np.ndarray
) -> np.ndarray:
    """Combine a strand's element resistances with electrically coupled
    resistive jackets in ideal parallel, THEA style.

    Jackets are not current carriers in the electric network; a nonzero
    ``electric_conductance_mode`` entry between a strand and a jacket in the
    coupling workbook marks the jacket as an ideal parallel conductor
    instead (zero transverse resistance). While the strand is
    superconducting its resistance is orders of magnitude below the
    jacket's, so the jacket share vanishes automatically; on quench the
    current commutes into the jacket in the ratio of the resistances.

    The per-element fraction of the total Joule power that belongs to the
    jacket (equal to its current fraction, R_parallel/R_jacket) is recorded
    in ``conductor.parallel_jacket_pairs`` for
    :func:`distribute_joule_power_to_parallel_jackets`.

    Args:
        conductor: Conductor object (coupling matrices and mesh available).
        strand: the current-carrying strand component.
        strand_resistance: per-element resistance of the strand alone in Ohm.

    Returns:
        np.ndarray: per-element resistance of the strand-jacket parallel.
    """
    pairs = [
        pair for pair in getattr(conductor, "parallel_jacket_pairs", [])
        if pair[0] is not strand
    ]
    effective_resistance = strand_resistance

    mode_matrix = conductor.coupling.electric_conductance_mode
    for jacket in conductor.inventory.jackets.collection:
        mode = ElectricConductanceMode.get_electric_conductance_mode(
            int(mode_matrix[strand.identifier, jacket.identifier])
        )
        if mode is ElectricConductanceMode.NO_CONDUCTANCE:
            continue
        jacket_resistance = (
            jacket.jacket_electrical_resistivity(jacket.gauss_fields)
            * conductor.mesh.element_lengths
            / (jacket.inputs.cross_section * jacket.inputs.cos_theta)
        )
        parallel_resistance = (
            effective_resistance * jacket_resistance
            / (effective_resistance + jacket_resistance)
        )
        # Fraction of the total current (and of the total Joule power) that
        # flows in the jacket.
        pairs.append(
            (strand, jacket, parallel_resistance / jacket_resistance)
        )
        effective_resistance = parallel_resistance

    conductor.parallel_jacket_pairs = pairs
    # get_joule_power_along reads the resistance from this attribute; keep it
    # consistent with the resistance matrix so the computed power is the
    # total power of the parallel pair.
    strand.gauss_fields.electric_resistance = effective_resistance
    return effective_resistance


def distribute_joule_power_to_parallel_jackets(conductor: object) -> None:
    """Split the Joule power of each strand-jacket ideal parallel.

    ``get_joule_power_along`` computes the strand's linear Joule power with
    the parallel (combined) resistance, i.e. the TOTAL power dissipated in
    the strand-jacket pair. This function moves each jacket's share
    (current fraction times total, since P_k = I_k^2 R_k) from the strand
    array to the jacket array, so that both components heat consistently
    with the current distribution.

    Must be called after every strand's and jacket's
    ``get_joule_power_along`` in the heat-source assembly.
    """
    pairs = getattr(conductor, "parallel_jacket_pairs", [])
    # The jackets' own get_joule_power_along never overwrites their power
    # array, so clear it before accumulating the shares of this step.
    for _, jacket, _ in pairs:
        jacket.gauss_fields.linear_power_el_resistance[:, 0] = 0.0
    for strand, jacket, jacket_fraction in pairs:
        total_power = strand.gauss_fields.linear_power_el_resistance[:, 0].copy()
        jacket.gauss_fields.linear_power_el_resistance[:, 0] += (
            total_power * jacket_fraction
        )
        strand.gauss_fields.linear_power_el_resistance[:, 0] = (
            total_power * (1.0 - jacket_fraction)
        )
        # Expose the jacket current for the output files (the strand's
        # current_along keeps the total current of the parallel pair).
        jacket.gauss_fields.current_along = (
            strand.gauss_fields.current_along * jacket_fraction
        )
