"""Thermal and electrical properties of aluminium alloy Al-6063 (W7-X jacket).

Translated from the CryoSoft/THEA Fortran property package ``Al6063.f``
(L. Bottura at CryoSoft, March 2013; K. Risse at IPP, February 2023) to
replicate the THEA W7-X conductor model with OPENSC2.
"""

import numpy as np


def density_al6063(temperature: np.ndarray) -> np.ndarray:
    """Density of Al alloy 6063 in kg/m^3.

    Constant value, valid for 0 <= T <= inf K.

    Reference:
        http://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MA6063T6
    """
    temperature = np.asarray(temperature)
    return 2560.0 * np.ones(temperature.shape)


def isobaric_specific_heat_al6063(temperature: np.ndarray) -> np.ndarray:
    """Specific heat of Al alloy 6063 in J/(kg K) for 1 <= T <= 1000 K.

    References:
        CryoComp Version 3.0
        V.J. Johnson, Properties of Materials at Low Temperatures (Phase 1),
        Pergamon Press, 1961
    """
    temperature_low_bound = 13.0300576
    temperature_high_bound = 54.4990277

    low_coefficients = (0.012034041, 0.039172353, 0.002543257, 0.000771961)
    mid_coefficients = (
        -4.797655015,
        1.249154424,
        -0.105581366,
        0.004593931,
        -3.65654e-05,
    )
    high_numerators = (9143.27476, -417.10708, 400.787649, -7722.6604)
    high_shifts = (-18.6481086, -25.7545827, -18.1481600, -5.44742024)
    high_exponents = (0.73861474, 1.52610058, 2.48698565, 3.70977399)

    temperature = np.clip(np.asarray(temperature, dtype=float), 1.0, 1000.0)

    intervals = [
        temperature <= temperature_low_bound,
        (temperature > temperature_low_bound)
        & (temperature <= temperature_high_bound),
        temperature > temperature_high_bound,
    ]
    behavior = [
        lambda tt: np.polynomial.polynomial.polyval(tt, low_coefficients),
        lambda tt: np.polynomial.polynomial.polyval(tt, mid_coefficients),
        lambda tt: sum(
            numerator * tt ** (order + 1) / (shift + tt) ** exponent
            for order, (numerator, shift, exponent) in enumerate(
                zip(high_numerators, high_shifts, high_exponents)
            )
        ),
    ]

    return np.piecewise(temperature, intervals, behavior)


def thermal_conductivity_al6063(temperature: np.ndarray) -> np.ndarray:
    """Thermal conductivity of Al alloy 6063 in W/(m K) for 2 <= T <= 1000 K.

    Reference:
        P. Reed & A.F. Clark, Materials at Low Temperature, ASM, 1983
    """
    k0 = 0.235849406
    k1 = 1.453488276
    k2 = -0.003823466
    km = 577.3864993
    temperature_peak = 5.282286082
    alpha = 1.030082721
    beta = 91.91507966
    exponent_n = 0.602305758
    exponent_m = 1.191691368

    temperature = np.clip(np.asarray(temperature, dtype=float), 2.0, 1000.0)

    reduced_temperature = temperature / temperature_peak
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


def electrical_resistivity_al6063(temperature: np.ndarray) -> np.ndarray:
    """Electrical resistivity of Al alloy 6063 (W7-X jacket) in Ohm m.

    SPC formula from report 1-AAB-T0152, based on measurements for
    4.2 <= T <= 300 K (evaluated over the clamped range 2 <= T <= 1000 K).
    """
    residual_resistivity = 8.20  # nOhm m
    p1 = 13.7
    p2 = 2.40
    p3 = 15.64
    p4 = 1.71

    temperature = np.clip(np.asarray(temperature, dtype=float), 2.0, 1000.0)

    intrinsic_resistivity = (
        p1
        * temperature ** p2
        / (1.0 + p3 * temperature ** p4 * np.exp(134.1 / temperature))
    )
    return (residual_resistivity + intrinsic_resistivity) * 1.0e-9
