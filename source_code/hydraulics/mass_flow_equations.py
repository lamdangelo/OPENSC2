"""Explicit (mdot, p, T) assembly of the 1D channel hydraulics.

Selected by ``explicit_mass_flow_formulation: true`` in the conductor inputs
(resolved formulation ``HydraulicFormulation.MASS_FLOW_EXPLICIT``).  The
Gauss-point blocks of the fluid unknowns W = (mdot, p, T), mdot = rho*A*v,
are assembled DIRECTLY from the written-out coefficients below, as an
independent route next to the similarity transform of
:mod:`hydraulics.formulation` (``mass_flow`` without the boolean).  The two
routes coincide for the flux and source blocks up to the thermodynamic
identities listed further down; they differ in the stabilisation block
(see :func:`build_kmat_fluid_mass_flow`).

Governing equations (quasi-linear, non-conservative form, per channel)

    d_t mdot + (2 mdot/(rho A)) d_x mdot + (A - kappa_T mdot^2/(rho A)) d_x p
             + (beta mdot^2/(rho A)) d_x T + (2 f |mdot|/(rho A D_h)) mdot = 0

    d_t p + (c^2/A) d_x mdot - ((gamma-1) mdot/(rho A)) d_x p
          + (beta c^2 mdot/A) d_x T - 2 f phi |mdot| mdot^2/(rho^2 A^3 D_h) = q_p

    d_t T + (phi T/(rho A)) d_x mdot - (phi T kappa_T mdot/(rho A)) d_x p
          + (gamma mdot/(rho A)) d_x T - 2 f |mdot| mdot^2/(rho^3 A^3 c_v D_h) = q_T

with v = mdot/(rho A) a DERIVED quantity, evaluated (like every coefficient)
at the state of the previous time level: the nodal velocity is rebuilt as
mdot/(rho A) from the native unknown after every step
(hydraulics.coolant._refresh_density_and_velocity_from_mass_flow) and
carried to the Gauss points by the same two-node average as p and T.  The
builders below read that Gauss velocity field rather than re-forming
mdot_G/(rho_G A) from the Gauss mass flow rate: the two differ by O(dz^2),
and the temperature-row exchange builders of thermal.energy_equation, which
are formulation-invariant and reused here, read the Gauss velocity field --
using anything else in the mdot and pressure rows would break the exact
row-to-row cancellations described further down.  Compact form
d_t W + K~ d_x W + S~ W = q~ with

    K~ = | 2v            A(1 - rho kappa_T v^2)   rho A beta v^2 |
         | c^2/A         -(gamma-1) v             rho beta c^2 v |
         | phi T/(rho A) -phi T kappa_T v         gamma v        |

    S~ = (2 f |v|/D_h) |  1                0  0 |
                       | -phi v / A        0  0 |
                       | -v/(rho A c_v)    0  0 |

Thermodynamic coefficients.  Direct from the property library (CoolProp
flash at the Gauss state): rho, beta, kappa_T, c_v, c_p, c (speed of sound).
Derived: gamma = c_p/c_v; phi = beta/(rho kappa_T c_v) (Gruneisen, the same
derivation the velocity path uses in hydraulics.compute_gruneisen).  The
coefficient listing above already uses the exact identities

    Reech:      rho c^2 kappa_T = gamma
    Mayer:      1 + phi beta T  = gamma

to write the diagonal advection entries as 2v, -(gamma-1)v and gamma v; with
the direct HEOS state these hold to roundoff (4e-16 measured), with the
tabulated property path (interfaces.coolprop_interface.USE_TABULAR_PROPERTIES,
independent bicubic splines per property) to ~1e-8 on 4-40 K / 3-10 bar
helium samples (tests/test_mass_flow_explicit.py; the table's own accuracy
degrades towards the pseudo-critical specific-heat peak), which is then the
level at which the spectrum of K~ equals {v-c, v, v+c} and at which this
route and the similarity transform differ per Gauss point.  No ideal-gas
relation is used anywhere.

Heat-exchange sources.  The fluid rows of the load vector are identically
zero: every source (friction, fluid-fluid and fluid-solid heat exchange, open
interface mass/momentum/energy transport) is an entry of the source-Jacobian
block S multiplying the state.  The pressure- and temperature-row exchange
entries satisfy q_p = phi rho c_v q_T column by column (they are built from
the same P*h and the same phi, rho, c_v), so the mdot row of the exchange
block vanishes identically for heat exchange, kappa_T q_p - beta q_T = 0.
The explicit route ENFORCES this structurally: the mdot row carries no
temperature column at all, and the only exchange term it carries is the
momentum +-K2 = +-K1 lambda_v v carried by the exchanged mass at open
interfaces.  :func:`check_source_consistency` asserts these facts on the
assembled block when the debug switch is on.
"""

import os
from typing import NamedTuple

import numpy as np

from hydraulics.hydraulics import compute_gruneisen

# Debug switch: OPENSC2_DEBUG_CHECKS=1 turns on the per-step source-block
# consistency assertion (read once at import so production runs pay nothing).
DEBUG_CHECKS_ENABLED = os.environ.get("OPENSC2_DEBUG_CHECKS", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Tolerances of check_source_consistency.
# Temperature columns: the identity is exact by construction (same phi, rho,
# c_v, P*h in both rows), so only roundoff is allowed.
TEMPERATURE_COLUMN_TOLERANCE = 1.0e-12
# Pressure columns of open interfaces: the residual kappa_T s_p - beta s_T is
# +-K1/(rho A) only through Reech + Mayer, which the tabulated property path
# satisfies to ~1e-8 away from the pseudo-critical peak (see module
# docstring); the tolerance leaves room for the peak region.
PRESSURE_COLUMN_TOLERANCE = 1.0e-4


class DerivedCoefficients(NamedTuple):
    """Per-Gauss-point coefficients of the explicit assembly.

    velocity: the Gauss velocity field, derived at the nodes as
    mdot/(rho A) from the native unknown (see module docstring);
    gamma = c_p/c_v (both direct); gruneisen = beta/(rho kappa_T c_v)."""
    velocity: np.ndarray
    gamma: np.ndarray
    gruneisen: np.ndarray


def thermodynamic_coefficients(fields, cross_section: float) -> DerivedCoefficients:
    """Derived coefficients at the Gauss points of one channel (see
    :class:`DerivedCoefficients`).  ``cross_section`` is accepted for
    signature symmetry with the builders (the velocity is not re-formed
    from the Gauss mass flow rate, see module docstring)."""
    velocity = np.asarray(fields.velocity, dtype=float)
    gamma = fields.total_isobaric_specific_heat / fields.total_isochoric_specific_heat
    gruneisen = compute_gruneisen(
        fields.isobaric_expansion_coefficient,
        fields.isothermal_compressibility,
        fields.total_isochoric_specific_heat,
        fields.total_density,
    )
    return DerivedCoefficients(velocity, gamma, gruneisen)


def build_amat_mass_flow(
    matrix: np.ndarray,
    f_comp,
    eq_idx: NamedTuple,
) -> np.ndarray:
    """Flux-Jacobian block K~ of one channel in (mdot, p, T) (see module
    docstring); explicit counterpart of
    utility_functions.step_matrix_construction.build_amat."""
    fields = f_comp.coolant.gauss_fields
    area = f_comp.channel.inputs.cross_section
    density = fields.total_density
    sound_speed_2 = fields.total_speed_of_sound ** 2
    beta = fields.isobaric_expansion_coefficient
    kappa_t = fields.isothermal_compressibility
    temperature = fields.temperature
    velocity, gamma, gruneisen = thermodynamic_coefficients(fields, area)

    # mdot equation.
    matrix[:, eq_idx.velocity, eq_idx.velocity] = 2.0 * velocity
    matrix[:, eq_idx.velocity, eq_idx.pressure] = area * (
        1.0 - density * kappa_t * velocity ** 2
    )
    matrix[:, eq_idx.velocity, eq_idx.temperature] = (
        density * area * beta * velocity ** 2
    )
    # pressure equation.
    matrix[:, eq_idx.pressure, eq_idx.velocity] = sound_speed_2 / area
    matrix[:, eq_idx.pressure, eq_idx.pressure] = -(gamma - 1.0) * velocity
    matrix[:, eq_idx.pressure, eq_idx.temperature] = (
        density * beta * sound_speed_2 * velocity
    )
    # temperature equation.
    matrix[:, eq_idx.temperature, eq_idx.velocity] = (
        gruneisen * temperature / (density * area)
    )
    matrix[:, eq_idx.temperature, eq_idx.pressure] = (
        -gruneisen * temperature * kappa_t * velocity
    )
    matrix[:, eq_idx.temperature, eq_idx.temperature] = gamma * velocity

    return matrix


def build_kmat_fluid_mass_flow(
    matrix: np.ndarray,
    upweqt: np.ndarray,
    f_comp,
    conductor,
) -> np.ndarray:
    """Upwind/stabilisation block of one channel in (mdot, p, T).

    The structure of the velocity-formulation builder
    (utility_functions.step_matrix_construction.build_kmat_fluid) is applied
    UNCHANGED to the (mdot, p, T) rows: diagonal artificial diffusion
    dz*u/2 with u the element-neighbourhood maximum of the nodal |v| (the
    stabilisation speed), and dz*(u + c)/2 on the mdot row (the acoustic
    characteristic speed), weighted by the same upwind_weights entries.

    TODO(stabilisation consistency): NOT ESTABLISHED.  The rows of the
    (mdot, p, T) system carry different dimensions than the (v, p, T) rows,
    and the similarity-transform route (hydraulics.formulation) conjugates
    this block, K' = M^-1 K M, which introduces off-diagonal entries
    (mdot-row p and T columns, p- and T-row mdot column) that the diagonal
    structure kept here does not have.  Neither route has been shown to be
    the consistent stabilisation of the other; the coefficients are
    deliberately NOT retuned.  The explicit-vs-transform tests measure the
    resulting solution difference and its convergence under refinement.
    """
    speed_of_sound = f_comp.coolant.gauss_fields.total_speed_of_sound
    delta_z = conductor.mesh.element_lengths
    eq_idx = conductor.equation_index[f_comp.identifier]

    # Stabilisation speed exactly as in build_kmat_fluid (nodal |v| of the
    # previous level, element max, spread one element to each side).
    node_speed = np.abs(np.ravel(f_comp.coolant.node_fields.velocity))
    element_speed = np.maximum(node_speed[:-1], node_speed[1:])
    padded_speed = np.pad(element_speed, 1, mode="edge")
    velocity = np.maximum(
        np.maximum(padded_speed[:-2], padded_speed[1:-1]), padded_speed[2:]
    )

    diag_idx = np.array(eq_idx)
    matrix[:, diag_idx, diag_idx] = (
        (delta_z * velocity / 2.0)[:, None] * upweqt[diag_idx][None, :]
    )
    matrix[:, eq_idx.velocity, eq_idx.velocity] = (
        delta_z * (velocity + speed_of_sound) / 2.0 * upweqt[eq_idx.velocity]
    )
    return matrix


def build_smat_fluid_momentum_mass_flow(
    matrix: np.ndarray,
    f_comp,
    eq_idx: NamedTuple,
) -> np.ndarray:
    """Friction entries of the mdot and pressure rows of S~ (mdot column):

        (mdot, mdot) = 2 f |mdot| / (rho A D_h)   [1/s]
        (p, mdot)    = -(2 f |v|/D_h) phi v / A   [Pa/kg]

    Semi-implicit friction: |mdot| (through |v|) frozen at the old level,
    mdot at the new level through the time integrator.  Must be called
    before :func:`build_smat_fluid_energy_mass_flow` on the same matrix."""
    fields = f_comp.coolant.gauss_fields
    area = f_comp.channel.inputs.cross_section
    velocity, _, gruneisen = thermodynamic_coefficients(fields, area)
    friction = (
        2.0
        * f_comp.channel.friction_factors[False].total
        * np.abs(velocity)
        / f_comp.channel.inputs.hydraulic_diameter
    )
    matrix[:, eq_idx.velocity, eq_idx.velocity] = friction
    matrix[:, eq_idx.pressure, eq_idx.velocity] = -friction * gruneisen * velocity / area
    return matrix


def build_smat_fluid_energy_mass_flow(
    matrix: np.ndarray,
    f_comp,
    eq_idx: NamedTuple,
) -> np.ndarray:
    """Frictional-heating entry of the temperature row of S~ (mdot column):

        (T, mdot) = -(2 f |v|/D_h) v / (rho A c_v)   [K/kg]

    Reuses the (mdot, mdot) friction entry written by
    :func:`build_smat_fluid_momentum_mass_flow`."""
    fields = f_comp.coolant.gauss_fields
    area = f_comp.channel.inputs.cross_section
    velocity, _, _ = thermodynamic_coefficients(fields, area)
    matrix[:, eq_idx.temperature, eq_idx.velocity] = (
        -matrix[:, eq_idx.velocity, eq_idx.velocity]
        * velocity
        / (fields.total_density * area * fields.total_isochoric_specific_heat)
    )
    return matrix


def build_smat_fluid_interface_momentum_mass_flow(
    matrix: np.ndarray,
    conductor,
) -> np.ndarray:
    """mdot- and pressure-row entries of S~ at fluid-fluid interfaces.

    Explicit counterpart of
    hydraulics.momentum_equation.build_smat_fluid_interface_momentum.  The
    temperature row is formulation-invariant on its p/T columns and is
    built by thermal.energy_equation.build_smat_fluid_interface_energy as
    in the velocity formulation."""
    eq_idx = conductor.equation_index
    for interface in conductor.interface.fluid_fluid:
        K1 = conductor.gauss_fields.K1[interface.interf_name]
        K2 = conductor.gauss_fields.K2[interface.interf_name]
        K3 = conductor.gauss_fields.K3[interface.interf_name]
        interf_peri = conductor.dict_interf_peri["ch_ch"]
        htc_gauss = conductor.gauss_fields.HTC["ch_ch"]
        coef_htc = (
            interf_peri["Open"]["Gauss"][interface.interf_name]
            * htc_gauss["Open"][interface.interf_name]
            + interf_peri["Close"]["Gauss"][interface.interf_name]
            * htc_gauss["Close"][interface.interf_name]
        )
        for comp_1, comp_2 in (
            (interface.comp_1, interface.comp_2),
            (interface.comp_2, interface.comp_1),
        ):
            matrix = _smat_fluid_interface_momentum_mass_flow(
                matrix, comp_1, comp_2, eq_idx, K1, K2, K3, coef_htc
            )
    return matrix


def _smat_fluid_interface_momentum_mass_flow(
    matrix: np.ndarray,
    comp_1,
    comp_2,
    eq_idx: dict,
    K1: np.ndarray,
    K2: np.ndarray,
    K3: np.ndarray,
    coef_htc,
) -> np.ndarray:
    """Rows of comp_1 (mdot and p), columns of comp_1 and comp_2.

    mdot row -- momentum carried by the exchanged mass only:

        (mdot_j, p_j) += +K2      (mdot_j, p_k) = -K2        [m^2]

    i.e. d_t mdot_j = ... - K2 (p_j - p_k) = -lambda_v v_donor Gamma with
    Gamma = K1 (p_j - p_k) the exchanged mass flow per unit length.  This is
    the closed form of the M^-1-combined (v, p, T) exchange rows: the
    energy-transport terms K3 - v K2 cancel exactly through
    kappa_T phi rho = beta/c_v, and the recovery terms c^2/phi and
    phi c_v T combine to rho c^2 kappa_T - beta phi T = 1 (Reech + Mayer).
    No temperature column: heat exchange does not act on mdot.

    p row -- unchanged from the velocity formulation (state columns p, T
    only, on which the secant P is the identity); the formulas are the ones
    of hydraulics.momentum_equation.__smat_fluid_interface_momentum:

        s_pj_pj = phi/A [K3 - v K2 - (w - v^2/2 - c^2/phi) K1]
        s_pj_tj = phi/A (P_o h_o + P_c h_c)
    """
    fields = comp_1.coolant.gauss_fields
    area = comp_1.channel.inputs.cross_section
    velocity, _, gruneisen = thermodynamic_coefficients(fields, area)
    j = eq_idx[comp_1.identifier]
    k = eq_idx[comp_2.identifier]

    # mdot row.
    matrix[:, j.velocity, j.pressure] += K2
    matrix[:, j.velocity, k.pressure] = -K2

    # pressure row.
    coef_grun_area = gruneisen / area
    s_pj_pj = coef_grun_area * (
        K3
        - velocity * K2
        - (
            fields.total_enthalpy
            - velocity ** 2 / 2.0
            - fields.total_speed_of_sound ** 2 / gruneisen
        )
        * K1
    )
    matrix[:, j.pressure, j.pressure] += s_pj_pj
    matrix[:, j.pressure, k.pressure] = -s_pj_pj
    s_pj_tj = coef_grun_area * coef_htc
    matrix[:, j.pressure, j.temperature] += s_pj_tj
    matrix[:, j.pressure, k.temperature] = -s_pj_tj
    return matrix


def check_source_consistency(
    source_jacobian: np.ndarray,
    conductor,
    temperature_tolerance: float = TEMPERATURE_COLUMN_TOLERANCE,
    pressure_tolerance: float = PRESSURE_COLUMN_TOLERANCE,
) -> dict:
    """Assert the heat-source consistency q_p = phi rho c_v q_T on the
    assembled (mdot, p, T) source-Jacobian block of every channel.

    Checked per channel j and per Gauss point:

    * every temperature column c (fluid or solid):
      |kappa_T S[p_j, c] - beta S[T_j, c]| <= tol * (|kappa_T S[p_j, c]| +
      |beta S[T_j, c]|), and S[mdot_j, c] == 0 exactly;
    * every pressure column of an interface partner k:
      kappa_T S[p_j, p_k] - beta S[T_j, p_k] == -K1/(rho A) and
      S[mdot_j, p_k] == -K2 (interface momentum), to pressure_tolerance
      (limited by the property identities, see module docstring).

    Returns the maximal relative residuals found (for reporting); raises
    AssertionError naming the channel, column and Gauss point otherwise.
    Called from the assembly when DEBUG_CHECKS_ENABLED; callable directly
    by tests."""
    eq_idx = conductor.equation_index
    temperature_columns = [
        eq_idx[f_comp.identifier].temperature
        for f_comp in conductor.inventory.fluids.collection
    ] + [eq_idx[s_comp.identifier] for s_comp in conductor.inventory.solids.collection]

    worst = {"temperature": 0.0, "pressure": 0.0, "mdot_row": 0.0}
    for f_comp in conductor.inventory.fluids.collection:
        j = eq_idx[f_comp.identifier]
        fields = f_comp.coolant.gauss_fields
        beta = fields.isobaric_expansion_coefficient
        kappa_t = fields.isothermal_compressibility
        for column in temperature_columns:
            a = kappa_t * source_jacobian[:, j.pressure, column]
            b = beta * source_jacobian[:, j.temperature, column]
            scale = np.abs(a) + np.abs(b)
            residual = np.abs(a - b)
            active = scale > 0.0
            if active.any():
                ratio = residual[active] / scale[active]
                worst["temperature"] = max(worst["temperature"], float(ratio.max()))
                bad = np.nonzero(ratio > temperature_tolerance)[0]
                assert bad.size == 0, (
                    f"{f_comp.identifier}: kappa_T q_p - beta q_T != 0 on "
                    f"temperature column {column} (relative residual "
                    f"{ratio[bad[0]]:.3e} at Gauss point {np.nonzero(active)[0][bad[0]]})"
                )
            mdot_row = source_jacobian[:, j.velocity, column]
            assert not np.any(mdot_row != 0.0), (
                f"{f_comp.identifier}: mdot row carries a temperature column "
                f"{column} (max |entry| {np.abs(mdot_row).max():.3e})"
            )

    for interface in conductor.interface.fluid_fluid:
        K1 = conductor.gauss_fields.K1[interface.interf_name]
        K2 = conductor.gauss_fields.K2[interface.interf_name]
        for comp_1, comp_2 in (
            (interface.comp_1, interface.comp_2),
            (interface.comp_2, interface.comp_1),
        ):
            j = eq_idx[comp_1.identifier]
            k = eq_idx[comp_2.identifier]
            fields = comp_1.coolant.gauss_fields
            rho_area = fields.total_density * comp_1.channel.inputs.cross_section
            expected = -K1 / rho_area
            found = (
                fields.isothermal_compressibility
                * source_jacobian[:, j.pressure, k.pressure]
                - fields.isobaric_expansion_coefficient
                * source_jacobian[:, j.temperature, k.pressure]
            )
            scale = np.abs(expected) + np.abs(found)
            active = scale > 0.0
            if active.any():
                ratio = np.abs(found - expected)[active] / scale[active]
                worst["pressure"] = max(worst["pressure"], float(ratio.max()))
                bad = np.nonzero(ratio > pressure_tolerance)[0]
                assert bad.size == 0, (
                    f"{comp_1.identifier}/{comp_2.identifier}: "
                    "kappa_T q_p - beta q_T != -K1/(rho A) on the partner "
                    f"pressure column (relative residual {ratio[bad[0]]:.3e})"
                )
            momentum = source_jacobian[:, j.velocity, k.pressure]
            scale = np.abs(K2) + np.abs(momentum)
            active = scale > 0.0
            if active.any():
                ratio = np.abs(momentum + K2)[active] / scale[active]
                worst["mdot_row"] = max(worst["mdot_row"], float(ratio.max()))
                assert not np.any(ratio > temperature_tolerance), (
                    f"{comp_1.identifier}/{comp_2.identifier}: mdot-row "
                    "interface entry is not -K2"
                )
    return worst
