"""NbTi properties, W7-X parameterization (CryoSoft/THEA W7-X model).

Faithful translation of the CryoSoft/THEA Fortran property package
``NbTi.f`` (L. Bottura at CryoSoft; G4 parameter set for W7-X, K. Risse
2023) to replicate the THEA W7-X conductor model with OPENSC2. Selected
with ``superconducting_material = NbTi-W7X`` in the strand input
workbook.

The critical surface is the Bottura practical fit (IEEE Trans. Appl.
Sup., 10(1), 1054-1057, 2000)::

    Jc(T, B) = C0/B * b^p * (1-b)^q * (1 - t^nu)^gamma

with ``t = T/Tc0``, ``b = B/Bc(T)`` and ``Bc(T) = Bc20 * (1 - t^nu)``.
``Tc0m``, ``Bc20m`` and ``C0`` come from the workbook (G4: 9.03 K,
14.61 T, 16.8512e10 A T/m^2, referred to the non-copper cross section);
the shape constants default to the G4 values.
"""

import numpy as np

# G4 parameter set (SPC report 1-AAB-T0152, K. Risse 2023).
FIT_CONSTANT_P = 1.0        # p (alpha) fitting constant of fp(b)
FIT_CONSTANT_Q = 1.54       # q (beta) fitting constant of fp(b)
FIT_CONSTANT_GAMMA = 2.1    # n (Gamma) fitting constant of h(t)
FIT_CONSTANT_NU = 1.7       # nu, fitting constant of Bc(T) and Tc(B)

_MAGNETIC_FIELD_LOW_BOUND = 1.0e-3  # T (Blow in the Fortran source)


def density_nbti_w7x(temperature: np.ndarray) -> np.ndarray:
    """Density of Nb-47%Ti in kg/m^3 (constant).

    Reference: Y. Iwasa, Case Studies in Superconducting Magnets, Springer.
    """
    temperature = np.asarray(temperature)
    return 6530.0 * np.ones(temperature.shape)


def thermal_conductivity_nbti_w7x(temperature: np.ndarray) -> np.ndarray:
    """Thermal conductivity of NbTi in W/(m K) for 1 <= T <= 400 K.

    References: EFDA Material Data Compilation (2007); Brechna (1973);
    Schmidt (1975). Continuous across the superconducting transition.
    """
    k0 = 0.020144234
    k1 = 0.0
    k2 = 3.624001366
    km = 1.25987251
    temperature_scale = 228.5582031
    alpha = 1.014972523
    beta = 0.132254649
    exponent_n = 1.364434093
    exponent_m = 1.457183987

    temperature = np.clip(np.asarray(temperature, dtype=float), 1.0, 400.0)

    reduced_temperature = temperature / temperature_scale
    return (
        k0
        + k1 * reduced_temperature
        + k2 * reduced_temperature ** 2
        + km
        * 3.0
        / (
            alpha * reduced_temperature ** exponent_n
            + beta / reduced_temperature ** exponent_m
        )
    )


def isobaric_specific_heat_nbti_w7x(
    temperature: np.ndarray,
    critical_temperature: np.ndarray,
    critical_temperature_at_0T: float,
) -> np.ndarray:
    """Specific heat of NbTi in J/(kg K) for 1 <= T <= 500 K.

    Normal state from a fit of Arp (1980) data matched to a Debye function;
    superconducting state from the specific-heat jump ``dC/dTc * Tc/Tc0``
    and a cubic temperature dependence. No allowance is made for the mixed
    state (Tcs < T < Tc), exactly as in the THEA source.

    Args:
        temperature: absolute temperature in K.
        critical_temperature: field-dependent critical temperature Tc(B) in K
            (as evaluated by :func:`critical_temperature_nbti_w7x`).
        critical_temperature_at_0T: Tc0m in K.
    """
    numerators = (83.31534936, 8.092610037, 17432.63, -17064.55061)
    shifts = (1793.124266, 1.893545785, 29.8873239, 23.30205048)
    exponents = (0.830308715, 8.615210509, 6.585468459, 8.596154557)
    specific_heat_jump_slope = 1.69

    temperature = np.clip(np.asarray(temperature, dtype=float), 1.0, 500.0)
    critical_temperature = np.asarray(critical_temperature, dtype=float)

    normal_specific_heat = sum(
        numerator * temperature ** exponent / (shift + temperature) ** exponent
        for numerator, shift, exponent in zip(numerators, shifts, exponents)
    )

    superconducting_specific_heat = (
        normal_specific_heat
        + specific_heat_jump_slope
        * critical_temperature
        / critical_temperature_at_0T
    ) * np.where(
        critical_temperature > 0.0,
        (temperature / np.maximum(critical_temperature, 1e-10)) ** 3,
        0.0,
    )

    return np.where(
        temperature <= critical_temperature,
        superconducting_specific_heat,
        normal_specific_heat,
    )


def electrical_resistivity_normal_nbti_w7x(
    temperature: np.ndarray,
) -> np.ndarray:
    """Resistivity of NbTi in the normal state in Ohm m for 10 <= T <= 1200 K.

    References: EFDA Material Data Compilation (2007); Milck (NTIS AD-750 579).
    """
    temperature = np.clip(np.asarray(temperature, dtype=float), 10.0, 1200.0)
    return 5.5668e-7 + 5.58e-10 * temperature


def critical_temperature_nbti_w7x(
    magnetic_field: np.ndarray,
    upper_critical_field_at_0K: float,
    critical_temperature_at_0T: float,
    nu: float = FIT_CONSTANT_NU,
) -> np.ndarray:
    """Critical temperature Tc(B) of NbTi in K (Lubell/Bottura fit).

    Tc = Tc0 * (1 - B/Bc20)^(1/nu), zero above Bc20.
    """
    magnetic_field = np.maximum(
        np.abs(np.asarray(magnetic_field, dtype=float)),
        _MAGNETIC_FIELD_LOW_BOUND,
    )
    reduced_field = magnetic_field / upper_critical_field_at_0K
    return critical_temperature_at_0T * np.where(
        reduced_field < 1.0,
        np.abs(1.0 - reduced_field) ** (1.0 / nu),
        0.0,
    )


def critical_magnetic_field_nbti_w7x(
    temperature: np.ndarray,
    upper_critical_field_at_0K: float,
    critical_temperature_at_0T: float,
    nu: float = FIT_CONSTANT_NU,
) -> np.ndarray:
    """Upper critical field Bc(T) of NbTi in T (Lubell/Bottura fit).

    Bc = Bc20 * (1 - (T/Tc0)^nu), zero above Tc0.
    """
    temperature = np.maximum(np.asarray(temperature, dtype=float), 0.0)
    reduced_temperature = temperature / critical_temperature_at_0T
    return upper_critical_field_at_0K * np.where(
        reduced_temperature < 1.0,
        1.0 - reduced_temperature ** nu,
        0.0,
    )


def critical_current_density_nbti_w7x(
    temperature: np.ndarray,
    magnetic_field: np.ndarray,
    upper_critical_field_at_0K: float,
    critical_current_scaling_constant: float,
    critical_temperature_at_0T: float,
    p: float = FIT_CONSTANT_P,
    q: float = FIT_CONSTANT_Q,
    gamma: float = FIT_CONSTANT_GAMMA,
    nu: float = FIT_CONSTANT_NU,
) -> np.ndarray:
    """Critical (non-copper) current density of NbTi in A/m^2 (Bottura fit).

    Jc = C0/B * b^p * (1-b)^q * (1 - t^nu)^gamma; zero at or above the
    critical temperature or the critical field. Argument order matches
    ``critical_current_density_nbti`` (Muzzi fit) for drop-in dispatch.
    """
    temperature = np.maximum(np.asarray(temperature, dtype=float), 0.0)
    magnetic_field = np.maximum(
        np.abs(np.asarray(magnetic_field, dtype=float)),
        _MAGNETIC_FIELD_LOW_BOUND,
    )
    temperature, magnetic_field = np.broadcast_arrays(
        np.atleast_1d(temperature), np.atleast_1d(magnetic_field)
    )

    reduced_temperature = temperature / critical_temperature_at_0T
    critical_field = critical_magnetic_field_nbti_w7x(
        temperature, upper_critical_field_at_0K, critical_temperature_at_0T, nu
    )

    critical_current_density = np.zeros_like(temperature)
    valid = (reduced_temperature < 1.0) & (magnetic_field < critical_field)
    if np.any(valid):
        reduced_field = magnetic_field[valid] / critical_field[valid]
        pinning_force_shape = reduced_field ** p * (1.0 - reduced_field) ** q
        temperature_shape = (
            1.0 - reduced_temperature[valid] ** nu
        ) ** gamma
        critical_current_density[valid] = (
            critical_current_scaling_constant
            / magnetic_field[valid]
            * pinning_force_shape
            * temperature_shape
        )
    return critical_current_density


def current_sharing_temperature_nbti_w7x(
    magnetic_field: np.ndarray,
    op_current_density: np.ndarray,
    upper_critical_field_at_0K: float,
    critical_current_scaling_constant: float,
    critical_temperature_at_0T: float,
    p: float = FIT_CONSTANT_P,
    q: float = FIT_CONSTANT_Q,
    gamma: float = FIT_CONSTANT_GAMMA,
    nu: float = FIT_CONSTANT_NU,
) -> np.ndarray:
    """Current sharing temperature of NbTi in K (inverse of the Jc fit).

    Vectorized bisection on the reduced temperature, equivalent to the
    Fortran ``TcsNbTi`` (tolerance 1e-5 on t = T/Tc0). Argument order
    matches ``current_sharing_temperature_nbti`` (Muzzi fit).
    """
    magnetic_field = np.atleast_1d(np.asarray(magnetic_field, dtype=float))
    op_current_density = np.atleast_1d(
        np.asarray(op_current_density, dtype=float)
    )[: magnetic_field.size]

    critical_temperature = critical_temperature_nbti_w7x(
        magnetic_field, upper_critical_field_at_0K, critical_temperature_at_0T, nu
    )

    def critical_current_density(temperature):
        return critical_current_density_nbti_w7x(
            temperature,
            magnetic_field,
            upper_critical_field_at_0K,
            critical_current_scaling_constant,
            critical_temperature_at_0T,
            p,
            q,
            gamma,
            nu,
        )

    # Bisection bounds on the reduced temperature t = T/Tc0.
    reduced_lower = np.zeros_like(magnetic_field)
    reduced_upper = critical_temperature / critical_temperature_at_0T

    # 40 halvings reach |t_up - t_low| < 1e-12, well below the Fortran
    # tolerance of 1e-5.
    for _ in range(40):
        reduced_middle = 0.5 * (reduced_lower + reduced_upper)
        above_operating = (
            critical_current_density(reduced_middle * critical_temperature_at_0T)
            > op_current_density
        )
        reduced_lower = np.where(above_operating, reduced_middle, reduced_lower)
        reduced_upper = np.where(above_operating, reduced_upper, reduced_middle)

    current_sharing_temperature = (
        0.5 * (reduced_lower + reduced_upper) * critical_temperature_at_0T
    )

    # Edge cases, exactly as in the Fortran source: field at or above Bc20
    # or operating current density at or above Jc(T=0) give zero; zero (or
    # negative) operating current density gives Tc(B).
    current_sharing_temperature = np.where(
        op_current_density <= 0.0,
        critical_temperature,
        current_sharing_temperature,
    )
    current_sharing_temperature = np.where(
        (magnetic_field >= upper_critical_field_at_0K)
        | (
            op_current_density
            >= critical_current_density(np.zeros_like(magnetic_field))
        ),
        0.0,
        current_sharing_temperature,
    )
    return current_sharing_temperature
