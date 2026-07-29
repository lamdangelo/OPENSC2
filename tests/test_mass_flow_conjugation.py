"""Unit tests of the (mdot, p, T) similarity-transform kernel.

The kernel must (i) preserve the eigenvalues of the fluid flux Jacobian -
a similarity transform cannot change them, and for the primitive-variable
Euler system they are analytically {v, v+c, v-c} - and (ii) coincide with
the dense reference T^-1 X T on general blocks, including two channels with
cross-channel (fluid-fluid interface) entries and a solid row/column, which
validates the two-phase (all rows, then all columns) ordering and the
invariance of solid rows.

States are drawn from a supercritical-helium-like envelope; the transform
coefficients are built exactly as the production wrapper builds them.
"""

import numpy as np
import pytest

from hydraulics.formulation import (
    conjugate_gauss_blocks,
    resolve_hydraulic_formulation,
)
from hydraulics.hydraulic_flags import (
    HydraulicFormulation,
    get_hydraulic_formulation,
)

RNG = np.random.default_rng(0)
NUMBER_OF_STATES = 200


def random_states(count):
    """Physically plausible supercritical-helium-like Gauss states."""
    return dict(
        density=RNG.uniform(40.0, 160.0, count),
        sound_speed=RNG.uniform(150.0, 400.0, count),
        velocity=RNG.uniform(-6.0, 6.0, count),
        gruneisen=RNG.uniform(0.2, 1.2, count),
        temperature=RNG.uniform(4.0, 20.0, count),
        isothermal_compressibility=10 ** RNG.uniform(-8.0, -5.0, count),
        isobaric_expansion_coefficient=10 ** RNG.uniform(-2.0, 0.0, count),
        cross_section=10 ** RNG.uniform(-5.0, -3.0, count),
    )


def flux_jacobian(states):
    """The fluid 3x3 flux Jacobian exactly as build_amat assembles it:
    diagonal v; (v,p) = 1/rho; (p,v) = rho c^2; (T,v) = Gruneisen * T."""
    count = states["velocity"].size
    matrix = np.zeros((count, 3, 3))
    matrix[:, 0, 0] = states["velocity"]
    matrix[:, 1, 1] = states["velocity"]
    matrix[:, 2, 2] = states["velocity"]
    matrix[:, 0, 1] = 1.0 / states["density"]
    matrix[:, 1, 0] = states["density"] * states["sound_speed"] ** 2
    matrix[:, 2, 0] = states["gruneisen"] * states["temperature"]
    return matrix


def transform_coefficients(states):
    rho_area = states["density"] * states["cross_section"]
    v_kappa = states["velocity"] * states["isothermal_compressibility"]
    v_beta = states["velocity"] * states["isobaric_expansion_coefficient"]
    return rho_area, v_kappa, v_beta


def dense_transform_matrices(states, index):
    """Dense 3x3 M and its analytic inverse for one state."""
    rho_area, v_kappa, v_beta = (
        coefficient[index] for coefficient in transform_coefficients(states)
    )
    forward = np.array(
        [
            [1.0 / rho_area, -v_kappa, v_beta],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    inverse = np.array(
        [
            [rho_area, rho_area * v_kappa, -rho_area * v_beta],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return forward, inverse


def test_forward_inverse_round_trip():
    states = random_states(NUMBER_OF_STATES)
    for index in range(NUMBER_OF_STATES):
        forward, inverse = dense_transform_matrices(states, index)
        residual = np.abs(forward @ inverse - np.eye(3)).max()
        assert residual < 1e-13 * np.abs(forward).max(), (
            f"state {index}: M M^-1 deviates from identity by {residual:.3e}"
        )


def test_eigenvalues_invariant_and_analytic():
    states = random_states(NUMBER_OF_STATES)
    original = flux_jacobian(states)
    transformed = original.copy()
    rho_area, v_kappa, v_beta = transform_coefficients(states)
    conjugate_gauss_blocks(
        [transformed], [(0, 1, 2, rho_area, v_kappa, v_beta)]
    )
    for index in range(NUMBER_OF_STATES):
        velocity = states["velocity"][index]
        sound_speed = states["sound_speed"][index]
        analytic = np.sort(
            [velocity, velocity + sound_speed, velocity - sound_speed]
        )
        eig_original = np.sort(np.linalg.eigvals(original[index]).real)
        eig_transformed = np.sort(np.linalg.eigvals(transformed[index]).real)
        scale = np.abs(analytic).max()
        assert np.abs(eig_original - analytic).max() < 1e-9 * scale
        assert np.abs(eig_transformed - eig_original).max() < 1e-9 * scale, (
            f"state {index}: eigenvalues not invariant under the transform"
        )


def test_kernel_matches_dense_reference_two_channels_one_solid():
    """Kernel vs dense T^-1 X T on full 7x7 blocks: channels at slots
    (0, 2, 4) and (1, 3, 5) - the production node-interleaved layout with
    two channels - plus a solid row/column at slot 6."""
    count = 50
    states_1 = random_states(count)
    states_2 = random_states(count)
    blocks = RNG.standard_normal((count, 7, 7))
    # Solid rows carry no fluid-velocity columns in the production system
    # (fluid-solid coupling is temperature-only): mirror that so the test
    # also demonstrates solid-row invariance under the transform.
    blocks[:, 6, 0] = 0.0
    blocks[:, 6, 1] = 0.0

    channels = [
        (0, 2, 4, *transform_coefficients(states_1)),
        (1, 3, 5, *transform_coefficients(states_2)),
    ]
    transformed = blocks.copy()
    conjugate_gauss_blocks([transformed], channels)

    for index in range(count):
        forward_1, _ = dense_transform_matrices(states_1, index)
        forward_2, _ = dense_transform_matrices(states_2, index)
        dense_forward = np.eye(7)
        for (v_idx, p_idx, t_idx), forward in (
            ((0, 2, 4), forward_1),
            ((1, 3, 5), forward_2),
        ):
            slots = np.ix_((v_idx, p_idx, t_idx), (v_idx, p_idx, t_idx))
            dense_forward[slots] = forward
        reference = (
            np.linalg.inv(dense_forward) @ blocks[index] @ dense_forward
        )
        deviation = np.abs(transformed[index] - reference).max()
        scale = np.abs(reference).max()
        assert deviation < 1e-12 * scale, (
            f"block {index}: kernel deviates from dense reference by "
            f"{deviation:.3e} (scale {scale:.3e})"
        )
        # Solid row is bit-invariant (no fluid-velocity columns).
        assert np.array_equal(transformed[index][6, 6], blocks[index][6, 6])


def test_solid_rows_and_load_free_slots_unchanged():
    """Rows without any velocity-column coupling must come out identical."""
    count = 20
    states = random_states(count)
    blocks = RNG.standard_normal((count, 4, 4))
    blocks[:, 3, 0] = 0.0  # solid row: no velocity column
    reference_solid_rows = blocks[:, 3, :].copy()
    conjugate_gauss_blocks(
        [blocks], [(0, 1, 2, *transform_coefficients(states))]
    )
    # Columns p, T of the solid row change only through its velocity column,
    # which is zero - the whole row must be bit-identical.
    assert np.array_equal(blocks[:, 3, :], reference_solid_rows)


# --------------------------------------------------------------------------
# Formulation resolution (pure function part)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared, coupling_enabled, expected",
    [
        (HydraulicFormulation.AUTO, False, HydraulicFormulation.VELOCITY),
        (HydraulicFormulation.AUTO, True, HydraulicFormulation.MASS_FLOW),
        (HydraulicFormulation.VELOCITY, False, HydraulicFormulation.VELOCITY),
        (HydraulicFormulation.MASS_FLOW, False, HydraulicFormulation.MASS_FLOW),
        (HydraulicFormulation.MASS_FLOW, True, HydraulicFormulation.MASS_FLOW),
    ],
)
def test_resolution_table(declared, coupling_enabled, expected):
    assert resolve_hydraulic_formulation(declared, coupling_enabled) is expected


def test_resolution_velocity_with_coupling_warns():
    with pytest.warns(UserWarning, match="lagged-density"):
        resolved = resolve_hydraulic_formulation(
            HydraulicFormulation.VELOCITY, True
        )
    assert resolved is HydraulicFormulation.VELOCITY


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("auto", HydraulicFormulation.AUTO),
        ("VELOCITY", HydraulicFormulation.VELOCITY),
        ("Mass_Flow", HydraulicFormulation.MASS_FLOW),
    ],
)
def test_get_hydraulic_formulation_accepts(flag, expected):
    assert get_hydraulic_formulation(flag) is expected


@pytest.mark.parametrize("flag", ["mdot", "", "velocity_formulation", 3])
def test_get_hydraulic_formulation_rejects(flag):
    with pytest.raises(ValueError, match="Unknown hydraulic formulation"):
        get_hydraulic_formulation(flag)
