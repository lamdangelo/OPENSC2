"""Selectable primary variables of the 1D channel hydraulics.

The solver assembles the quasilinear compressible flow system in the
primitive variables W = (v, p, T) per channel. The alternative
W' = (mdot, p, T) formulation (mdot = rho*A*v, the native unknown of the
hydraulic-network coupling) is obtained by a similarity transform of the
already-assembled Gauss-point matrices instead of a second set of
hand-derived PDE coefficients, so the two formulations cannot drift apart.

With M = dW/dW' the change-of-variables Jacobian, frozen at the same state
the matrices were built from (the standard Picard linearization of the
solver),

        | 1/(rho A)   -v kappa_T   v beta |            | rho A   rho A v kappa_T   -rho A v beta |
    M = |     0            1          0   | ,   M^-1 = |   0            1                 0      |
        |     0            0          1   |            |   0            0                 1      |

(kappa_T isothermal compressibility, beta isobaric expansion coefficient),
the transformed system is A' = M^-1 A M per Gauss point for the blocks that
act on state DERIVATIVES (flux Jacobian and upwind/diffusion - the chain
rule dW = M dW' is exact); eigenvalues {v, v+c, v-c} are invariant.  The
source-Jacobian block multiplies the STATE and transforms as
S' = M^-1 S P with the Picard secant P = diag(1/(rho A), 1, 1) instead
(see :func:`transform_source_block` for why the Jacobian is wrong there).
The fluid transient block is the identity and M^-1 I M = I, so the
mass-capacity matrix is untouched; the fluid rows of the load vector are
identically zero, so it is untouched as well.

M and M^-1 differ from the identity only in the mdot-row/v-column of each
channel triple, so the global transform reduces to one row operation and
one column operation per channel. The left phase must complete for all
channels before the right phase starts: fluid-fluid interface blocks couple
two channels' triples, and the column phase of channel j reads entries the
row phase of channel i must already have transformed.
"""

import warnings

from hydraulics.hydraulic_flags import HydraulicFormulation


def resolve_hydraulic_formulation(
    declared: HydraulicFormulation, coupling_enabled: bool
) -> HydraulicFormulation:
    """Resolve the declared (input) formulation to an executable one.

    AUTO becomes MASS_FLOW when the conductor is coupled to the hydraulic
    network (the port variable is then a native unknown), VELOCITY
    otherwise. An explicit choice always wins; explicitly requesting the
    velocity formulation on a coupled conductor is honoured with a warning,
    since the port mass flow is then only enforced through the lagged-
    density Picard linearization.
    """
    if declared is HydraulicFormulation.AUTO:
        if coupling_enabled:
            return HydraulicFormulation.MASS_FLOW
        return HydraulicFormulation.VELOCITY
    if declared is HydraulicFormulation.VELOCITY and coupling_enabled:
        warnings.warn(
            "Velocity formulation explicitly requested on a conductor "
            "coupled to the hydraulic network: the port mass flow is "
            "enforced through the lagged-density linearization instead of "
            "a native unknown. Consider hydraulic_formulation: mass_flow."
        )
    return declared


def conjugate_gauss_blocks(matrices, channels) -> None:
    """In-place similarity transform X -> M^-1 X M of Gauss-point blocks.

    Args:
        matrices: iterable of arrays shaped (n_elements, ndf, ndf) - the
            per-Gauss-point flux-Jacobian, diffusion and source-Jacobian
            blocks (the mass-capacity block is invariant and must not be
            passed).
        channels: iterable of tuples
            (velocity_index, pressure_index, temperature_index,
             rho_area, v_kappa, v_beta)
            with the three DOF slots of one channel inside the ndf block
            and the coefficient arrays shaped (n_elements,):
            rho_area = rho * A, v_kappa = v * kappa_T, v_beta = v * beta.
    """
    channels = [
        (v_idx, p_idx, t_idx, rho_area[:, None], v_kappa[:, None], v_beta[:, None])
        for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels
    ]
    for matrix in matrices:
        # Left phase (M^-1 X): only the mdot row of each channel changes.
        for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels:
            matrix[:, v_idx, :] = rho_area * (
                matrix[:, v_idx, :]
                + v_kappa * matrix[:, p_idx, :]
                - v_beta * matrix[:, t_idx, :]
            )
        # Right phase (X M): only the (v, p, T) columns of each channel
        # change, reading the already left-transformed v column.
        for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels:
            column_v = matrix[:, :, v_idx].copy()
            matrix[:, :, p_idx] -= v_kappa[:, 0][:, None] * column_v
            matrix[:, :, t_idx] += v_beta[:, 0][:, None] * column_v
            matrix[:, :, v_idx] = column_v / rho_area[:, 0][:, None]


def transform_source_block(matrix, channels) -> None:
    """In-place transform S -> M^-1 S P of the Gauss-point source block.

    The source term multiplies the STATE, not its derivative.  The chain
    rule that makes M^-1 X M exact for the flux and diffusion blocks does
    not apply here: Phi (the value map W = Phi(W')) is not degree-1
    homogeneous, so the Jacobian M does NOT reproduce it --
    (M W')_v = v * (1 + beta*T - kappa_T*p) != v at the very state M was
    built from.  Conjugating S with M therefore evaluates the source at
    that parasitic state, multiplying the effective channel friction by
    (1 + beta*T - kappa_T*p) ~ 2.5 for supercritical helium at 6 K
    (measured as an 18-27 % steady flow deficit on the AAB51 deck).  The
    right factor must be the Picard SECANT P = diag(1/(rho*A), 1, 1),
    which satisfies P W' = Phi(W') exactly at the frozen state; the left
    factor M^-1 (a recombination of the equations) is unchanged.  Do not
    "simplify" this back to a similarity transform.

    Same argument/shape conventions as :func:`conjugate_gauss_blocks`;
    the left phase runs over all channels before the right phase for the
    same fluid-fluid interface reason.
    """
    channels = [
        (v_idx, p_idx, t_idx, rho_area[:, None], v_kappa[:, None], v_beta[:, None])
        for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels
    ]
    # Left phase (M^-1 S): only the mdot row of each channel changes.
    for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels:
        matrix[:, v_idx, :] = rho_area * (
            matrix[:, v_idx, :]
            + v_kappa * matrix[:, p_idx, :]
            - v_beta * matrix[:, t_idx, :]
        )
    # Right phase (S P): only the v column of each channel changes, scaled
    # by its own 1/(rho*A); the p and T columns of P are the identity.
    for v_idx, p_idx, t_idx, rho_area, v_kappa, v_beta in channels:
        matrix[:, :, v_idx] = matrix[:, :, v_idx] / rho_area[:, 0][:, None]


def transform_gauss_matrices_to_mass_flow(gauss_point_matrices, conductor) -> None:
    """Transform the assembled Gauss-point matrices to (mdot, p, T) unknowns.

    Called once per step by ``assemble_thermal_hydraulic_system`` when the
    conductor's resolved hydraulic formulation is MASS_FLOW, after every
    Gauss-point builder has run and before the element-matrix construction.
    Coefficients are the Gauss-point fields the untransformed matrices were
    built from, so the transform is an exact similarity per Gauss point.
    """
    channels = []
    for f_comp in conductor.inventory.fluids.collection:
        eq_idx = conductor.equation_index[f_comp.identifier]
        fields = f_comp.coolant.gauss_fields
        cross_section = f_comp.channel.inputs.cross_section
        channels.append(
            (
                eq_idx.velocity,
                eq_idx.pressure,
                eq_idx.temperature,
                fields.total_density * cross_section,
                fields.velocity * fields.isothermal_compressibility,
                fields.velocity * fields.isobaric_expansion_coefficient,
            )
        )
    # Derivative-acting blocks transform by the chain rule (similarity
    # with the Jacobian M); the state-multiplying source block transforms
    # with the Picard secant P on the right (see transform_source_block).
    conjugate_gauss_blocks(
        (
            gauss_point_matrices.flux_jacobian,
            gauss_point_matrices.diffusion,
        ),
        channels,
    )
    transform_source_block(gauss_point_matrices.source_jacobian, channels)
