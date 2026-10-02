"""Tier-4 reference: manufactured solution and the numerical residual of the
code's fluid PDE system in either formulation.

Manufactured fields (single channel of length L, period tau):

    mdot_M(x,t) = mdot_0 [1 + 0.2 sin(2 pi x/L) cos(2 pi t/tau)]
    p_M(x,t)    = p_0 - dp x/L + 0.02 p_0 sin(pi x/L) sin(2 pi t/tau)
    T_M(x,t)    = T_0 + dT sin(pi x/L)^2 (1 - cos(2 pi t/tau))/2

The residual R = LHS(W_M) of the PDE system of the chosen formulation is
computed NUMERICALLY: derivatives by central finite differences of the
manufactured fields (relative step 1e-6) and thermodynamic coefficients from
the injected property function at (p_M, T_M); the derived fields v = mdot/(rho A)
and rho are differentiated as composed functions, so the property variation
is included. Added as a source to the solver, W_M becomes the exact solution.

Velocity formulation rows (units m/s^2, Pa/s, K/s):
    R_v = v_t + v v_x + p_x/rho + F v
    R_p = p_t + rho c^2 v_x + v p_x - F phi rho v^2
    R_T = T_t + phi T v_x + v T_x - F v^2 / c_v
Mass-flow formulation rows (kg/s^2, Pa/s, K/s):
    R_m = mdot_t + 2 v mdot_x + A (1 - rho kappa_T v^2) p_x + rho A beta v^2 T_x + F mdot
    R_p = p_t + (c^2/A) mdot_x - (gamma - 1) v p_x + rho beta c^2 v T_x - F phi v mdot / A
    R_T = T_t + (phi T/(rho A)) mdot_x - phi T kappa_T v p_x + gamma v T_x - F v mdot/(rho A c_v)
with F = 2 f(Re) |v| / D_h, gamma = c_p/c_v, phi = beta/(rho kappa_T c_v).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ManufacturedSolution:
    mass_flow_0: float
    pressure_0: float
    pressure_drop: float
    length: float
    period: float
    temperature_0: float = 4.5
    temperature_amplitude: float = 2.0
    mass_flow_relative_amplitude: float = 0.2
    pressure_relative_amplitude: float = 0.02

    def mass_flow(self, x, t):
        return self.mass_flow_0 * (
            1.0 + self.mass_flow_relative_amplitude
            * np.sin(2 * np.pi * x / self.length) * np.cos(2 * np.pi * t / self.period)
        )

    def pressure(self, x, t):
        return (
            self.pressure_0 - self.pressure_drop * x / self.length
            + self.pressure_relative_amplitude * self.pressure_0
            * np.sin(np.pi * x / self.length) * np.sin(2 * np.pi * t / self.period)
        )

    def temperature(self, x, t):
        return self.temperature_0 + self.temperature_amplitude * np.sin(
            np.pi * x / self.length
        ) ** 2 * (1.0 - np.cos(2 * np.pi * t / self.period)) / 2.0


@dataclass
class VelocityManufacturedSolution(ManufacturedSolution):
    """Tier-5 variant: the VELOCITY is the prescribed sinusoid and the mass
    flow is the derived field,

        v_M(x,t)    = v_0 [1 + a sin(2 pi x/L) cos(2 pi t/tau)],
        mdot_M(x,t) = rho(p_M, T_M) A v_M(x,t),

    with v_0 = mdot_0 / (rho(p_0, T_0) A) so that the inlet mass flow, the
    base state and the boundary values coincide with the Tier-4 solution.
    p_M and T_M are unchanged. ``properties`` is the (T, p) -> dict property
    callable of the solver and ``area`` the channel cross-section; both are
    required because the derived mass flow needs the density."""
    properties: object = None
    area: float = None

    @property
    def velocity_0(self) -> float:
        density_0 = _properties_vector(self.properties, self.temperature_0, self.pressure_0)["density"]
        return self.mass_flow_0 / (float(density_0[0]) * self.area)

    def velocity(self, x, t):
        return self.velocity_0 * (
            1.0 + self.mass_flow_relative_amplitude
            * np.sin(2 * np.pi * x / self.length) * np.cos(2 * np.pi * t / self.period)
        )

    def mass_flow(self, x, t):
        density = _properties_vector(self.properties, self.temperature(x, t), self.pressure(x, t))["density"]
        return density * self.area * self.velocity(x, t)


def _properties_vector(properties, temperature, pressure) -> dict:
    """Vectorised property evaluation through the scalar/array callable."""
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))
    result = properties(temperature, pressure)
    return {key: np.asarray(value, dtype=float) for key, value in result.items()}


def derived_fields(solution: ManufacturedSolution, x, t, properties, area):
    """rho, v and the property set at the manufactured state."""
    pressure = solution.pressure(x, t)
    temperature = solution.temperature(x, t)
    props = _properties_vector(properties, temperature, pressure)
    velocity = solution.mass_flow(x, t) / (props["density"] * area)
    return pressure, temperature, props, velocity


def residual(solution: ManufacturedSolution, x, t, formulation: str, properties,
             friction_factor, area: float, hydraulic_diameter: float,
             relative_step: float = 1.0e-6) -> dict:
    """Residual rows of the chosen formulation ('velocity' or 'mass_flow') at
    points x (array) and time t (scalar)."""
    x = np.asarray(x, dtype=float)
    hx = relative_step * solution.length
    ht = relative_step * solution.period

    def fields(xx, tt):
        p, T, props, v = derived_fields(solution, xx, tt, properties, area)
        return dict(mdot=solution.mass_flow(xx, tt), p=p, T=T, v=v)

    centre = fields(x, t)
    plus_x, minus_x = fields(x + hx, t), fields(x - hx, t)
    plus_t, minus_t = fields(x, t + ht), fields(x, t - ht)
    d_x = {k: (plus_x[k] - minus_x[k]) / (2 * hx) for k in centre}
    d_t = {k: (plus_t[k] - minus_t[k]) / (2 * ht) for k in centre}

    p, T, props, v = derived_fields(solution, x, t, properties, area)
    mdot = centre["mdot"]
    rho = props["density"]
    kappa = props["isothermal_compressibility"]
    beta = props["isobaric_expansion_coefficient"]
    cv = props["isochoric_specific_heat"]
    cp = props["isobaric_specific_heat"]
    c2 = props["speed_of_sound"] ** 2
    mu = props["viscosity"]
    phi = beta / (rho * kappa * cv)
    gamma = cp / cv
    reynolds = np.abs(v) * rho * hydraulic_diameter / mu
    F = 2.0 * np.asarray(friction_factor(reynolds), dtype=float) * np.abs(v) / hydraulic_diameter

    if formulation == "velocity":
        return {
            "velocity": d_t["v"] + v * d_x["v"] + d_x["p"] / rho + F * v,
            "pressure": d_t["p"] + rho * c2 * d_x["v"] + v * d_x["p"] - F * phi * rho * v ** 2,
            "temperature": d_t["T"] + phi * T * d_x["v"] + v * d_x["T"] - F * v ** 2 / cv,
        }
    if formulation == "mass_flow":
        return {
            "velocity": (
                d_t["mdot"] + 2 * v * d_x["mdot"] + area * (1 - rho * kappa * v ** 2) * d_x["p"]
                + rho * area * beta * v ** 2 * d_x["T"] + F * mdot
            ),
            "pressure": (
                d_t["p"] + c2 / area * d_x["mdot"] - (gamma - 1) * v * d_x["p"]
                + rho * beta * c2 * v * d_x["T"] - F * phi * v * mdot / area
            ),
            "temperature": (
                d_t["T"] + phi * T / (rho * area) * d_x["mdot"] - phi * T * kappa * v * d_x["p"]
                + gamma * v * d_x["T"] - F * v * mdot / (rho * area * cv)
            ),
        }
    raise ValueError(f"unknown formulation {formulation!r}")
