"""Tier-1 reference: steady flow split of N parallel channels between two
lumped volumes, each volume fed from a fixed-pressure reservoir through a
linear valve.

The steady state of the code's continuous channel model (quasi-linear
(v, p, T) system with friction 2 f(Re) |v| v / D_h, no heating) is the ODE
system in x, for a given mass flow mdot = rho A v (constant along the channel):

    v' = -v (kappa_T p' - beta T')                       (continuity, rho(p,T))
    v v' + p'/rho + 2 f |v| v / D_h = 0                  (momentum)
    v T' + phi T v' - 2 f |v| v^2 / (c_v D_h) = 0        (energy, phi Grueneisen)

which is a 2x2 linear system for (p', T') at every x:

    [ 1/rho - v^2 kappa_T     v^2 beta      ] [p']   [ -F v      ]
    [ -phi T kappa_T          1 + phi T beta ] [T'] = [  F v / c_v ]      F = 2 f |v| / D_h.

The isothermal-incompressible limit reduces to Darcy-Weisbach
dp = 2 f (L/D_h) mdot |mdot| / (rho A^2); for helium at 4.5 K / 5 bar the
density varies by ~1 % over a 0.5 bar drop and the frictional/expansion
temperature change is ~0.1 K, so the ODE (not the scalar equation) is the
exact steady state of the continuous model. Axial heat conduction in the
solid components (which equilibrate to the local fluid temperature) is
neglected: its relative effect on the flow is bounded by
(k_s A_s / L) / (mdot c_p) ~ 1e-4 of the ~0.1 K temperature variation, i.e.
~1e-6 on the density.

Network closure: p_supply_volume = p_supply - R_supply Q,
p_return_volume = p_return + R_return Q with Q = sum_k mdot_k (linear valves,
dp = R mdot), the port continuity p_channel_end = p_volume being exact in the
coupled model. Nested scalar root finding (brentq) to 1e-14 relative.

Injected callables:
    properties(temperature: float, pressure: float) -> dict with keys
        density, isothermal_compressibility, isobaric_expansion_coefficient,
        isochoric_specific_heat, viscosity   (SI)
    friction_factor(reynolds: float) -> float   total Fanning friction factor
        (including any multiplier), as the code evaluates it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass
class ChannelSpec:
    identifier: str
    length: float  # m
    cross_section: float  # m^2
    hydraulic_diameter: float  # m
    friction_factor: Callable[[float], float]


@dataclass
class ChannelProfile:
    x: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    mass_flow: float


@dataclass
class SteadyNetworkSolution:
    mass_flows: np.ndarray  # per channel [kg/s]
    total_mass_flow: float
    pressure_supply_volume: float
    pressure_return_volume: float
    profiles: list = field(default_factory=list)  # ChannelProfile per channel
    root_residual: float = 0.0


def gruneisen(props: dict) -> float:
    beta = props["isobaric_expansion_coefficient"]
    if beta == 0.0:
        return 0.0
    return beta / (
        props["density"]
        * props["isothermal_compressibility"]
        * props["isochoric_specific_heat"]
    )


def steady_gradients(pressure, temperature, mass_flow, channel: ChannelSpec,
                     properties) -> tuple:
    """(dp/dx, dT/dx) of the steady channel ODE at one point."""
    props = properties(float(temperature), float(pressure))
    rho = props["density"]
    kappa = props["isothermal_compressibility"]
    beta = props["isobaric_expansion_coefficient"]
    cv = props["isochoric_specific_heat"]
    mu = props["viscosity"]
    phi = gruneisen(props)
    v = mass_flow / (rho * channel.cross_section)
    reynolds = abs(v) * rho * channel.hydraulic_diameter / mu
    friction = 2.0 * channel.friction_factor(reynolds) * abs(v) / channel.hydraulic_diameter
    matrix = np.array(
        [
            [1.0 / rho - v * v * kappa, v * v * beta],
            [-phi * temperature * kappa, 1.0 + phi * temperature * beta],
        ]
    )
    rhs = np.array([-friction * v, friction * v / cv])
    dp_dx, dt_dx = np.linalg.solve(matrix, rhs)
    return dp_dx, dt_dx, v, rho


def integrate_channel(mass_flow: float, pressure_inlet: float,
                      temperature_inlet: float, channel: ChannelSpec,
                      properties, rtol: float = 1.0e-12,
                      profile_points: int = 0) -> ChannelProfile:
    """Integrate the steady ODE from x = 0 to x = L for a given mass flow."""

    def rhs(x, y):
        dp_dx, dt_dx, _, _ = steady_gradients(y[0], y[1], mass_flow, channel, properties)
        return [dp_dx, dt_dx]

    t_eval = (
        np.linspace(0.0, channel.length, profile_points) if profile_points else None
    )
    solution = solve_ivp(
        rhs,
        (0.0, channel.length),
        [pressure_inlet, temperature_inlet],
        method="DOP853",
        rtol=rtol,
        atol=[1e-9 * pressure_inlet, 1e-12],
        t_eval=t_eval,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(f"steady channel integration failed: {solution.message}")
    x = solution.t
    pressure = solution.y[0]
    temperature = solution.y[1]
    density = np.array(
        [properties(float(T), float(p))["density"] for T, p in zip(temperature, pressure)]
    )
    velocity = mass_flow / (density * channel.cross_section)
    return ChannelProfile(x, pressure, temperature, density, velocity, mass_flow)


def channel_mass_flow(pressure_inlet: float, pressure_outlet: float,
                      temperature_inlet: float, channel: ChannelSpec,
                      properties, xtol_relative: float = 1.0e-14,
                      rtol_ode: float = 1.0e-12) -> float:
    """Mass flow of one channel for given end pressures (root of
    p_out(mdot) - pressure_outlet), bracketed from the incompressible
    Darcy-Weisbach estimate."""
    drop = pressure_inlet - pressure_outlet
    if drop <= 0.0:
        raise ValueError("channel_mass_flow expects pressure_inlet > pressure_outlet")
    props = properties(float(temperature_inlet), float(pressure_inlet))
    rho = props["density"]
    mu = props["viscosity"]
    # Darcy-Weisbach fixed point for the bracket centre.
    mass_flow = 1.0e-3
    for _ in range(100):
        v = mass_flow / (rho * channel.cross_section)
        reynolds = abs(v) * rho * channel.hydraulic_diameter / mu
        f = channel.friction_factor(reynolds)
        v_new = np.sqrt(drop * channel.hydraulic_diameter / (2.0 * f * rho * channel.length))
        mass_flow_new = rho * channel.cross_section * v_new
        if abs(mass_flow_new - mass_flow) < 1e-12 * mass_flow_new:
            break
        mass_flow = mass_flow_new
    estimate = mass_flow_new

    def residual(m):
        return integrate_channel(m, pressure_inlet, temperature_inlet, channel,
                                 properties, rtol=rtol_ode).pressure[-1] - pressure_outlet

    lower, upper = 0.5 * estimate, 1.5 * estimate
    # Expand the bracket if needed (p_out decreases with mdot).
    for _ in range(20):
        if residual(lower) > 0.0 and residual(upper) < 0.0:
            break
        lower *= 0.5
        upper *= 1.5
    return brentq(residual, lower, upper, xtol=xtol_relative * estimate,
                  rtol=4.0 * np.finfo(float).eps, maxiter=200)


def solve_steady_network(channels: Sequence[ChannelSpec], pressure_supply: float,
                         pressure_return: float, temperature_supply: float,
                         resistance_supply: float, resistance_return: float,
                         properties, xtol_relative: float = 1.0e-14,
                         rtol_ode: float = 1.0e-12,
                         profile_points: int = 401) -> SteadyNetworkSolution:
    """Steady flow split of parallel channels between the two volume nodes."""

    def split(total):
        p_sv = pressure_supply - resistance_supply * total
        p_rv = pressure_return + resistance_return * total
        flows = np.array(
            [
                channel_mass_flow(p_sv, p_rv, temperature_supply, channel, properties,
                                  xtol_relative, rtol_ode)
                for channel in channels
            ]
        )
        return flows, p_sv, p_rv

    def residual(total):
        flows, _, _ = split(total)
        return flows.sum() - total

    if resistance_supply == 0.0 and resistance_return == 0.0:
        flows, p_sv, p_rv = split(0.0)
        total = float(flows.sum())
        root_residual = 0.0
    else:
        flows0, _, _ = split(0.0)
        guess = float(flows0.sum())
        # Upper bracket: the valve drops must leave a positive channel drop.
        crossing = (pressure_supply - pressure_return) / (resistance_supply + resistance_return)
        upper = min(1.05 * guess, (1.0 - 1.0e-9) * crossing)
        total = brentq(residual, 0.0, upper, xtol=xtol_relative * guess,
                       rtol=4.0 * np.finfo(float).eps, maxiter=200)
        flows, p_sv, p_rv = split(total)
        root_residual = float(abs(flows.sum() - total) / total)
    profiles = [
        integrate_channel(m, p_sv, temperature_supply, channel, properties,
                          rtol=rtol_ode, profile_points=profile_points)
        for m, channel in zip(flows, channels)
    ]
    return SteadyNetworkSolution(
        mass_flows=flows,
        total_mass_flow=float(flows.sum()),
        pressure_supply_volume=p_sv,
        pressure_return_volume=p_rv,
        profiles=profiles,
        root_residual=root_residual,
    )
