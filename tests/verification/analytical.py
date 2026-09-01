"""Closed-form analytical reference solutions for the verification suite.

Pure numpy functions, no OPENSC2 imports. Each docstring states the formula
and its literature reference:

- V1: linear network theory (Kirchhoff laws), any circuit-theory text.
- V2/V3: E. B. Wylie, V. L. Streeter, "Fluid Transients in Systems",
  Prentice Hall, 1993 — mass-oscillation (surge tank) and water-hammer
  (Joukowsky) chapters.
- V4: standard transmission-line theory (reflection at a resistive
  termination), e.g. Wylie & Streeter ch. 3 or any microwave-network text.
"""

from __future__ import annotations

import numpy as np


# --- Fluid / element parameters ---------------------------------------------


def laminar_pipe_resistance(
    viscosity: float,
    length: float,
    diameter: float,
    density: float,
    cross_section: float,
) -> float:
    """Hagen-Poiseuille resistance of a pipe in (pressure, mass-flow)
    variables.

    dp = 32 mu l v / D^2 (laminar pipe flow; Wylie & Streeter eq. 2-11 with
    Fanning f = 16/Re) and mdot = rho A v give

        R = dp/mdot = 32 mu l / (D^2 rho A)   [Pa/(kg/s)]

    which is exactly the OPENSC2 pipe-branch resistance
    2 f l |mdot| / (D rho A^2) with f = 16/Re, Re = |mdot| D / (A mu),
    independent of the flow magnitude (hence also of the solver's
    LAMINAR_REYNOLDS_FLOOR).
    """
    return 32.0 * viscosity * length / (diameter**2 * density * cross_section)


def pipe_inertance(length: float, cross_section: float) -> float:
    """Hydraulic inertance L = l/A [1/m] of a pipe in (p, mdot) variables:
    (l/A) dmdot/dt = dp (rigid-column theory; Wylie & Streeter ch. 2)."""
    return length / cross_section


def hydraulic_capacitance(
    volume: float, density: float, isothermal_compressibility: float
) -> float:
    """Node storage C = V rho kappa_T [kg/Pa]: dm = V d(rho) = V rho kappa_T dp
    at constant temperature — the OPENSC2 node capacitance."""
    return volume * density * isothermal_compressibility


def speed_of_sound(density: float, isothermal_compressibility: float) -> float:
    """Acoustic wave speed a = 1/sqrt(rho kappa) [m/s] (Wylie & Streeter
    eq. 1-6 for a rigid conduit). With the constant-property fluid's beta = 0
    the isentropic and isothermal compressibilities coincide, so this is the
    exact wave speed of the channel model."""
    return 1.0 / np.sqrt(density * isothermal_compressibility)


def characteristic_impedance(wave_speed: float, cross_section: float) -> float:
    """Characteristic impedance Z = a / A [Pa/(kg/s)] of a fluid line in
    (p, mdot) variables.

    From the linearized line equations in (p, mdot),

        (1/A) dmdot/dt = -dp/dx        (inertance per length l = 1/A)
        (A rho kappa) dp/dt = -dmdot/dx  (capacitance per length c = A rho kappa)

    Z = sqrt(l/c) = 1/(A sqrt(rho kappa)) = a/A: the density cancels.
    Equivalently, Joukowsky dp = rho a dv (Wylie & Streeter eq. 1-1) with
    dmdot = rho A dv gives dp/dmdot = a/A. (The familiar rho a / A is the
    impedance in (p, volume-flow) variables.) Dimensional check:
    a/A ~ (m/s)/m^2 = 1/(m s) = Pa/(kg/s).
    """
    return wave_speed / cross_section


# --- V1: steady Kirchhoff ---------------------------------------------------


def steady_network_reference(
    internal_incidence: np.ndarray,
    reservoir_incidence: np.ndarray,
    resistances: np.ndarray,
    reservoir_pressures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Steady state of a linear resistive network (Kirchhoff laws).

    With oriented incidence N (+1 leaving / -1 entering) split into internal
    rows A_int and reservoir rows A_res, branch conductances G = diag(1/R):

        branch flow      mdot = G (N^T p) with p over all nodes, i.e.
                         mdot = -G (A_int^T p_int + A_res^T p_res)
        node balance     A_int mdot = 0

    (the sign follows the OPENSC2 branch momentum R mdot = p_from - p_to and
    N[from] = +1, so N^T p = p_from - p_to enters with +; flows leave through
    +1 entries). Eliminating mdot gives the reduced nodal system

        (A_int G A_int^T) p_int = -(A_int G A_res^T) p_res

    Returns (internal node pressures, branch mass flows).
    """
    conductance = np.diag(1.0 / resistances)
    lhs = internal_incidence @ conductance @ internal_incidence.T
    rhs = -internal_incidence @ conductance @ reservoir_incidence.T @ reservoir_pressures
    internal_pressures = np.linalg.solve(lhs, rhs)
    branch_flows = conductance @ (
        internal_incidence.T @ internal_pressures
        + reservoir_incidence.T @ reservoir_pressures
    )
    return internal_pressures, branch_flows


# --- V2: series RLC free response -------------------------------------------


def rlc_parameters(
    resistance: float, inertance: float, capacitance: float
) -> tuple[float, float, float]:
    """Series-RLC parameters of the surge-tank mass oscillation
    (Wylie & Streeter, surge-tank chapter; L dq/dt + R q + (1/C) int q = 0):

        sigma   = R / (2 L)              decay rate [1/s]
        omega_0 = 1 / sqrt(L C)          undamped angular frequency [rad/s]
        omega_d = sqrt(omega_0^2 - sigma^2)  damped angular frequency

    Returns (sigma, omega_0, omega_d).
    """
    sigma = resistance / (2.0 * inertance)
    omega_0 = 1.0 / np.sqrt(inertance * capacitance)
    omega_d = np.sqrt(omega_0**2 - sigma**2)
    return sigma, omega_0, omega_d


def rlc_free_response(
    time: np.ndarray, pressure_offset: float, sigma: float, omega_d: float
) -> np.ndarray:
    """Underdamped free response of the node pressure, initial conditions
    p(0) - p_res = dp0 and mdot(0) = 0 (zero initial flow means
    dp/dt(0) = 0 in the node storage equation):

        p(t) - p_res = dp0 e^{-sigma t} [cos(omega_d t)
                                         + (sigma/omega_d) sin(omega_d t)]

    Wylie & Streeter, surge-tank oscillation with friction (linearized).
    """
    return (
        pressure_offset
        * np.exp(-sigma * time)
        * (np.cos(omega_d * time) + (sigma / omega_d) * np.sin(omega_d * time))
    )


# --- V3: water hammer -------------------------------------------------------


def closed_end_square_wave(
    time: np.ndarray,
    steady_pressure: float,
    joukowsky_rise: float,
    wave_period: float,
) -> np.ndarray:
    """Exact frictionless water-hammer pressure at the closed end of a line
    fed by a constant-pressure reservoir after instantaneous closure at
    t = 0 (Wylie & Streeter ch. 1-2, the classical valve-closure staircase):

        p(t) = p0 + dp   for t in [0, T/2) mod T
        p(t) = p0 - dp   for t in [T/2, T) mod T

    with dp = Z mdot0 (Joukowsky) and T = 4L/a. This square wave contains
    exactly the odd organ-pipe harmonics f_n = (2n-1) a / (4L).
    """
    phase = np.mod(time, wave_period)
    return np.where(
        phase < 0.5 * wave_period,
        steady_pressure + joukowsky_rise,
        steady_pressure - joukowsky_rise,
    )


def organ_pipe_frequencies(wave_speed: float, length: float, count: int) -> np.ndarray:
    """Eigenfrequencies of a line with a pressure node at one end (reservoir)
    and a pressure antinode at the other (closed end), quarter-wave resonator:

        f_n = (2n - 1) a / (4 L),  n = 1..count

    Wylie & Streeter ch. 8 (resonance in pipe systems).
    """
    n = np.arange(1, count + 1)
    return (2 * n - 1) * wave_speed / (4.0 * length)


# --- V4: transmission-line reflection ---------------------------------------


def reflection_coefficient(termination_resistance: float, impedance: float) -> float:
    """Pressure-wave reflection coefficient of a lumped resistive termination
    on a line of characteristic impedance Z (standard transmission-line
    theory; Wylie & Streeter ch. 3):

        Gamma = (R_term - Z) / (R_term + Z)

    Gamma = -1 for a pressure reservoir (R -> 0), +1 for a closed end
    (R -> inf), 0 for the matched termination R = Z.
    """
    return (termination_resistance - impedance) / (termination_resistance + impedance)


def raised_cosine_pulse(
    time: np.ndarray, start: float, width: float, amplitude: float
) -> np.ndarray:
    """Raised-cosine (Hann) pulse of the given full width:

        s(t) = A/2 [1 - cos(2 pi (t - t0)/w)]  for t in [t0, t0 + w], else 0.

    Compact support and continuous first derivative — band-limited enough to
    propagate cleanly on a mesh resolving w with many elements.
    """
    phase = (time - start) / width
    pulse = 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * phase))
    return np.where((phase >= 0.0) & (phase <= 1.0), pulse, 0.0)


# --- V5: first-order thermal lag --------------------------------------------


def first_order_lag(
    time: np.ndarray,
    initial_temperature: float,
    inflow_temperature: float,
    rate: float,
) -> np.ndarray:
    """Well-mixed volume flushed by a constant inflow (M cp dT/dt =
    mdot_in cp (T_in - T), cp constant):

        T(t) = T_in + (T0 - T_in) e^{-rate t},  rate = mdot_in / M,
        M = rho0 V.
    """
    return inflow_temperature + (
        initial_temperature - inflow_temperature
    ) * np.exp(-rate * time)


# --- Signal-processing helpers ----------------------------------------------


def fft_peak_frequencies(
    signal: np.ndarray,
    time_step: float,
    count: int,
    zero_padding: int = 8,
) -> np.ndarray:
    """Frequencies of the `count` largest local maxima of the amplitude
    spectrum, refined by parabolic interpolation on log-amplitude.

    The signal is detrended (mean removed) and Hann-windowed to suppress
    leakage from the finite record, zero-padded `zero_padding` times for a
    dense spectral grid, and the three-point parabolic fit around each peak
    bin removes most of the remaining bin-quantization error. Returns the
    peak frequencies sorted ascending.
    """
    samples = np.asarray(signal, dtype=float)
    samples = samples - samples.mean()
    window = np.hanning(samples.size)
    padded_size = zero_padding * samples.size
    spectrum = np.abs(np.fft.rfft(samples * window, n=padded_size))
    frequencies = np.fft.rfftfreq(padded_size, d=time_step)

    interior = np.arange(1, spectrum.size - 1)
    is_peak = (spectrum[interior] > spectrum[interior - 1]) & (
        spectrum[interior] >= spectrum[interior + 1]
    )
    peak_bins = interior[is_peak]
    if peak_bins.size < count:
        raise ValueError(
            f"Found only {peak_bins.size} spectral peaks, need {count}."
        )
    largest = peak_bins[np.argsort(spectrum[peak_bins])[-count:]]

    refined = []
    bin_width = frequencies[1] - frequencies[0]
    for peak_bin in largest:
        left, center, right = np.log(spectrum[peak_bin - 1 : peak_bin + 2])
        offset = 0.5 * (left - right) / (left - 2.0 * center + right)
        refined.append(frequencies[peak_bin] + offset * bin_width)
    return np.sort(np.asarray(refined))


def fitted_order(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    """Observed convergence order: least-squares slope of log(error) against
    log(step size) over a step-halving sequence."""
    return float(
        np.polyfit(np.log(np.asarray(step_sizes)), np.log(np.asarray(errors)), 1)[0]
    )


def relative_l2_error(computed: np.ndarray, reference: np.ndarray) -> float:
    """Discrete relative L2 error ||computed - reference|| / ||reference||."""
    reference = np.asarray(reference, dtype=float)
    return float(
        np.linalg.norm(np.asarray(computed, dtype=float) - reference)
        / np.linalg.norm(reference)
    )


def current_sharing_reference(
    total_current: float,
    critical_current: float,
    power_law_exponent: float,
    electric_field_criterion: float,
    stack_stabilizer_area: float,
    external_stabilizer_area: float,
    copper_resistivity: float,
) -> dict:
    """Uniform parallel-circuit current split with a power-law SC branch.

    All paths share the same longitudinal electric field E (ideal
    transverse coupling, equipotential terminations):

        I_sc(E)  = Ic * (E / E0)^(1/n)
        I_st(E)  = E * A_st  / rho_cu   (stabilizer inside the stack)
        I_ext(E) = E * A_ext / rho_cu   (external copper strand)
        I_sc + I_st + I_ext = I0

    Solved for E by bisection; returns the per-branch currents. Valid
    for I0 > Ic (dissipative regime) with temperature- and
    field-constant copper resistivity.
    """
    from scipy.optimize import brentq

    def total(field: float) -> float:
        sc = critical_current * (
            field / electric_field_criterion
        ) ** (1.0 / power_law_exponent)
        stabilizer = field * stack_stabilizer_area / copper_resistivity
        external = field * external_stabilizer_area / copper_resistivity
        return sc + stabilizer + external - total_current

    field = brentq(total, 1e-12, 1e3, xtol=1e-18, rtol=1e-14)
    sc_current = critical_current * (
        field / electric_field_criterion
    ) ** (1.0 / power_law_exponent)
    stack_stabilizer_current = (
        field * stack_stabilizer_area / copper_resistivity
    )
    external_current = field * external_stabilizer_area / copper_resistivity
    return {
        "electric_field": field,
        "sc_current": sc_current,
        "stack_current": sc_current + stack_stabilizer_current,
        "external_current": external_current,
    }
