"""Tier-2 reference: linear transmission-line model of N channels between two
lumped volumes, solved with a high-order method independent of the main
solver.

Each channel is the linearisation of the code's continuous (v, p) equations
about the Tier-1 steady state (x-dependent base state rho_b, c_b, v_b):

    d_t dv = -v_b d_x dv - (1/rho_b) d_x dp - F'(v_b) dv
    d_t dp = -rho_b c_b^2 d_x dv - v_b d_x dp + W(x) dv

with F(v) = 2 f(Re(v)) |v| v / D_h the friction acceleration (F' its tangent,
evaluated numerically from the code's own friction law by the caller) and
W = -(dp_b/dx) - G'(v_b) the pressure-row coefficient of dv, G(v) = F(v) phi
rho v the frictional-heating term of the pressure equation. The temperature
perturbation is slaved (isentropic acoustics): C' = A/c_b^2 with c_b the
library speed of sound, i.e. rho c^2 kappa_T = gamma. Mass flow perturbation
dmdot = rho_b A dv + A v_b dp / c_b^2. In (p, mdot) variables and without
mean flow this is the classical lossy line L' = 1/A, C' = A/c^2,
R' = F'(v_b)/A; note that the task's R' = 2 f |v_b| / D_h omits the
tangent's Re-dependence of f (for f = a/Re + b the Darcy term contributes
once, not twice) and the 1/A.

Terminations (volume nodes): C dP/dt = (P_res(t) - P)/R_far - sum_k dmdot_k(0)
(supply) and C dP/dt = sum_k dmdot_k(L_k) - P/R_far (return), C = V rho kappa_T
as the code's HydraulicNetwork defines it; C = 0 makes the node algebraic
(P = P_res - R_far sum dmdot), as in the code. Optional node thermal model of
the code (mixing enthalpy balance with the adiabatic port inflow temperature
and the expansion source V rho beta dT/dt).

Numerics: interior points advanced with 4th-order central finite differences
(biased 4th-order stencils next to the ends) and classical RK4 at CFL 0.1;
the line ends are closed by CHARACTERISTICS: the outgoing Riemann invariant
(p + rho c v at x = L, p - rho c v at x = 0) is extrapolated (cubic) from the
interior, the incoming one follows from the termination (node pressure, or
the algebraic node relation), so that stiff terminations (R -> 0 or R -> inf,
C -> 0) never restrict the time step. The caller refines the grid until the
reported change falls below tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class LineBase:
    """Base state of one line on the reference grid (uniform x)."""
    identifier: str
    x: np.ndarray
    area: float
    density: np.ndarray
    sound_speed: np.ndarray
    velocity: np.ndarray
    friction_tangent: np.ndarray  # F'(v_b) [1/s]
    pressure_work: np.ndarray  # W(x) [Pa/m]

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    def mass_flow(self, dv, dp):
        return self.area * (self.density * dv + self.velocity * dp / self.sound_speed ** 2)


@dataclass
class NodeTermination:
    capacitance: float  # kg/Pa (0 -> algebraic node)
    far_resistance: float  # Pa/(kg/s)
    reservoir: Callable[[float], float] = lambda t: 0.0  # dP_res(t) [Pa]
    # Optional node thermal model (code's enthalpy balance + expansion source).
    thermal: bool = False
    volume: float = 0.0
    density: float = 0.0
    expansion_coefficient: float = 0.0
    inflow_rate: float = 0.0  # kg/s steady inflow through the node
    inflow_temperature_gain: float = 0.0  # dT_in / dp_port  [K/Pa], adiabatic


@dataclass
class LineSolution:
    times: np.ndarray
    node_pressure_supply: np.ndarray
    node_pressure_return: np.ndarray
    port_flow_inlet: dict  # identifier -> dmdot(0, t)
    port_flow_outlet: dict  # identifier -> dmdot(L, t)
    snapshot_times: np.ndarray
    snapshots: dict = field(default_factory=dict)  # identifier -> {"x", "pressure", "mass_flow"}
    grid_points: dict = field(default_factory=dict)
    time_step: float = 0.0


# 4th-order first-derivative stencils.
_CENTRAL = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
_FORWARD = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 12.0  # at i using i..i+4
_BIASED = np.array([-3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0  # at i using i-1..i+3
# Cubic extrapolation to the boundary from the four nearest interior points
# (Lagrange weights of points at distances 1, 2, 3, 4 grid steps).
_EXTRAPOLATE = np.array([4.0, -6.0, 4.0, -1.0])


def derivative(u: np.ndarray, dx: float) -> np.ndarray:
    du = np.empty_like(u)
    du[2:-2] = (_CENTRAL[0] * u[:-4] + _CENTRAL[1] * u[1:-3] + _CENTRAL[3] * u[3:-1]
                + _CENTRAL[4] * u[4:])
    du[0] = np.dot(_FORWARD, u[:5])
    du[1] = np.dot(_BIASED, u[:5])
    du[-1] = -np.dot(_FORWARD, u[-5:][::-1])
    du[-2] = -np.dot(_BIASED, u[-5:][::-1])
    return du / dx


class _System:
    """Line + node system with the characteristic boundary closure."""

    def __init__(self, lines, supply, return_node):
        self.lines = list(lines)
        self.supply = supply
        self.return_node = return_node

    # -- boundary closure --------------------------------------------------
    def close(self, dv, dp, nodes, t):
        """Set the boundary values of every line and the algebraic node
        pressures from the extrapolated outgoing invariants."""
        # Outgoing invariants at the ends.
        w_minus_0 = []  # p - rho c v at x = 0 (left-going)
        w_plus_L = []  # p + rho c v at x = L (right-going)
        for line, v, p in zip(self.lines, dv, dp):
            zc = line.density * line.sound_speed
            w_minus = p - zc * v
            w_plus = p + zc * v
            w_minus_0.append(float(np.dot(_EXTRAPOLATE, w_minus[1:5])))
            w_plus_L.append(float(np.dot(_EXTRAPOLATE, w_plus[-5:-1][::-1])))
        # Node pressures.
        if self.supply.capacitance > 0.0:
            P_s = nodes[0]
        else:
            # P_s = P_res - R sum_k mdot_k(0), mdot_k(0) = A [rho v_0 + vbar p_0 / c^2]
            # with p_0 = P_s and v_0 = (P_s - w-) / (rho c).
            gain = sum(
                line.area * (1.0 / line.sound_speed[0] + line.velocity[0] / line.sound_speed[0] ** 2)
                for line in self.lines
            )
            drive = sum(
                line.area * w / line.sound_speed[0] for line, w in zip(self.lines, w_minus_0)
            )
            P_s = (self.supply.reservoir(t) + self.supply.far_resistance * drive) / (
                1.0 + self.supply.far_resistance * gain
            )
            nodes[0] = P_s
        if self.return_node.capacitance > 0.0:
            P_r = nodes[1]
        else:
            gain = sum(
                line.area * (1.0 / line.sound_speed[-1] - line.velocity[-1] / line.sound_speed[-1] ** 2)
                for line in self.lines
            )
            drive = sum(
                line.area * w / line.sound_speed[-1] for line, w in zip(self.lines, w_plus_L)
            )
            P_r = self.return_node.far_resistance * drive / (
                1.0 + self.return_node.far_resistance * gain
            )
            nodes[1] = P_r
        # Boundary states.
        for line, v, p, w_m, w_p in zip(self.lines, dv, dp, w_minus_0, w_plus_L):
            p[0] = P_s
            v[0] = (P_s - w_m) / (line.density[0] * line.sound_speed[0])
            p[-1] = P_r
            v[-1] = (w_p - P_r) / (line.density[-1] * line.sound_speed[-1])

    # -- right-hand side ---------------------------------------------------
    def rhs(self, dv, dp, nodes, t):
        d_dv, d_dp = [], []
        inflow_supply = 0.0
        inflow_return = 0.0
        for line, v, p in zip(self.lines, dv, dp):
            dx_v = derivative(v, line.dx)
            dx_p = derivative(p, line.dx)
            dvdt = -line.velocity * dx_v - dx_p / line.density - line.friction_tangent * v
            dpdt = (-line.density * line.sound_speed ** 2 * dx_v - line.velocity * dx_p
                    + line.pressure_work * v)
            # Boundary points are algebraic (closed after each stage).
            dvdt[0] = dvdt[-1] = 0.0
            dpdt[0] = dpdt[-1] = 0.0
            d_dv.append(dvdt)
            d_dp.append(dpdt)
            flow = line.mass_flow(v, p)
            inflow_supply -= flow[0]
            inflow_return += flow[-1]
        P_s, P_r, T_s, T_r = nodes
        dT_s = dT_r = 0.0
        source_s = source_r = 0.0
        supply, return_node = self.supply, self.return_node
        if supply.thermal and supply.volume > 0.0:
            dT_s = -supply.inflow_rate / (supply.density * supply.volume) * T_s
            source_s = supply.volume * supply.density * supply.expansion_coefficient * dT_s
        if return_node.thermal and return_node.volume > 0.0:
            dT_in = return_node.inflow_temperature_gain * P_r
            dT_r = return_node.inflow_rate / (return_node.density * return_node.volume) * (dT_in - T_r)
            source_r = return_node.volume * return_node.density * return_node.expansion_coefficient * dT_r
        dP_s = 0.0
        if supply.capacitance > 0.0:
            dP_s = ((supply.reservoir(t) - P_s) / supply.far_resistance + inflow_supply + source_s) / supply.capacitance
        dP_r = 0.0
        if return_node.capacitance > 0.0:
            dP_r = (inflow_return - P_r / return_node.far_resistance + source_r) / return_node.capacitance
        return d_dv, d_dp, np.array([dP_s, dP_r, dT_s, dT_r])


def solve_transmission_line(lines: Sequence[LineBase], supply: NodeTermination,
                            return_node: NodeTermination, t_end: float,
                            snapshot_times: Sequence[float], cfl: float = 0.1,
                            series_stride: int = 1) -> LineSolution:
    """Integrate the linear line + node system from rest to t_end."""
    system = _System(lines, supply, return_node)
    lines = system.lines
    dt = cfl * min(
        line.dx / float(np.max(line.sound_speed + np.abs(line.velocity))) for line in lines
    )
    number_of_steps = int(np.ceil(t_end / dt))
    dt = t_end / number_of_steps
    snapshot_times = np.asarray(sorted(snapshot_times), dtype=float)
    snapshot_steps = set(np.rint(snapshot_times / dt).astype(int).tolist())

    dv = [np.zeros_like(line.x) for line in lines]
    dp = [np.zeros_like(line.x) for line in lines]
    nodes = np.zeros(4)  # P_s, P_r, dT_s, dT_r
    system.close(dv, dp, nodes, 0.0)

    def stage(base, derivative_state, factor, t_stage):
        dv_b, dp_b, nodes_b = base
        d_dv, d_dp, d_nodes = derivative_state
        new_dv = [v + factor * dvv for v, dvv in zip(dv_b, d_dv)]
        new_dp = [p + factor * dpp for p, dpp in zip(dp_b, d_dp)]
        new_nodes = nodes_b + factor * d_nodes
        system.close(new_dv, new_dp, new_nodes, t_stage)
        return new_dv, new_dp, new_nodes

    times = [0.0]
    P_s_series = [nodes[0]]
    P_r_series = [nodes[1]]
    inlet_series = {line.identifier: [0.0] for line in lines}
    outlet_series = {line.identifier: [0.0] for line in lines}
    snapshots = {line.identifier: {"pressure": [], "mass_flow": []} for line in lines}
    recorded_snapshot_times = []
    state = (dv, dp, nodes)
    t = 0.0
    for step in range(1, number_of_steps + 1):
        k1 = system.rhs(*state, t)
        s2 = stage(state, k1, 0.5 * dt, t + 0.5 * dt)
        k2 = system.rhs(*s2, t + 0.5 * dt)
        s3 = stage(state, k2, 0.5 * dt, t + 0.5 * dt)
        k3 = system.rhs(*s3, t + 0.5 * dt)
        s4 = stage(state, k3, dt, t + dt)
        k4 = system.rhs(*s4, t + dt)
        combined = (
            [(a + 2 * b + 2 * c + d) / 6.0 for a, b, c, d in zip(k1[0], k2[0], k3[0], k4[0])],
            [(a + 2 * b + 2 * c + d) / 6.0 for a, b, c, d in zip(k1[1], k2[1], k3[1], k4[1])],
            (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )
        t = step * dt
        state = stage(state, combined, dt, t)
        if step % series_stride == 0 or step == number_of_steps:
            times.append(t)
            P_s_series.append(state[2][0])
            P_r_series.append(state[2][1])
            for line, v, p in zip(lines, state[0], state[1]):
                flow = line.mass_flow(v, p)
                inlet_series[line.identifier].append(flow[0])
                outlet_series[line.identifier].append(flow[-1])
        if step in snapshot_steps:
            recorded_snapshot_times.append(t)
            for line, v, p in zip(lines, state[0], state[1]):
                snapshots[line.identifier]["pressure"].append(p.copy())
                snapshots[line.identifier]["mass_flow"].append(line.mass_flow(v, p))
    result = LineSolution(
        times=np.asarray(times),
        node_pressure_supply=np.asarray(P_s_series),
        node_pressure_return=np.asarray(P_r_series),
        port_flow_inlet={k: np.asarray(v) for k, v in inlet_series.items()},
        port_flow_outlet={k: np.asarray(v) for k, v in outlet_series.items()},
        snapshot_times=np.asarray(recorded_snapshot_times),
        time_step=dt,
    )
    for line in lines:
        result.snapshots[line.identifier] = {
            "x": line.x.copy(),
            "pressure": np.array(snapshots[line.identifier]["pressure"]),
            "mass_flow": np.array(snapshots[line.identifier]["mass_flow"]),
        }
        result.grid_points[line.identifier] = line.x.size
    return result


def uniform_line(identifier: str, length: float, points: int, area: float,
                 density: float, sound_speed: float, velocity: float = 0.0,
                 friction_tangent: float = 0.0, pressure_work: float = 0.0) -> LineBase:
    """Constant-coefficient line (unit tests and quick estimates)."""
    x = np.linspace(0.0, length, points)
    ones = np.ones_like(x)
    return LineBase(identifier, x, area, density * ones, sound_speed * ones,
                    velocity * ones, friction_tangent * ones, pressure_work * ones)


def raised_cosine_ramp(amplitude: float, duration: float):
    """Smooth step 0 -> amplitude over [0, duration] (C1), then constant."""

    def schedule(t):
        if t <= 0.0:
            return 0.0
        if t >= duration:
            return amplitude
        return 0.5 * amplitude * (1.0 - np.cos(np.pi * t / duration))

    return schedule


def smooth_step(amplitude: float, duration: float):
    """C-infinity step 0 -> amplitude over [0, duration] (bump-function
    smoothstep s = f(u)/(f(u) + f(1 - u)), f(u) = exp(-1/u)): every derivative
    vanishes at both ends, so a high-order scheme converges at its nominal
    order. Used for the Tier-2 reservoir step in both the reference and the
    solver runs."""

    def bump(u):
        return np.exp(-1.0 / u) if u > 0.0 else 0.0

    def schedule(t):
        if t <= 0.0:
            return 0.0
        if t >= duration:
            return amplitude
        u = t / duration
        return amplitude * bump(u) / (bump(u) + bump(1.0 - u))

    return schedule


def sinusoid(amplitude: float, angular_frequency: float, ramp_duration: float):
    """Sinusoidal drive switched on smoothly over ramp_duration."""
    envelope = smooth_step(1.0, ramp_duration)

    def schedule(t):
        return amplitude * envelope(t) * np.sin(angular_frequency * t)

    return schedule


def propagation_constant(angular_frequency: float, sound_speed: float,
                         friction_tangent: float) -> complex:
    """gamma = alpha + i beta of the uniform lossy line d_t v = -p_x/rho - F v,
    d_t p = -rho c^2 v_x: p ~ exp(-gamma x + i omega t) with
    gamma^2 = -(omega^2 - i omega F)/c^2."""
    return np.sqrt(-(angular_frequency ** 2) + 1j * angular_frequency * friction_tangent + 0j) / sound_speed


def reflection_coefficient_volume(impedance_line: float, far_resistance: float,
                                  capacitance: float, angular_frequency: float) -> complex:
    """Analytic reflection coefficient of a line of impedance Z terminated by
    a volume capacitance C in parallel with the far-side resistance R:
    Z_vol = 1 / (1/R + i omega C), Gamma = (Z_vol - Z)/(Z_vol + Z)."""
    z_vol = 1.0 / (1.0 / far_resistance + 1j * angular_frequency * capacitance)
    return (z_vol - impedance_line) / (z_vol + impedance_line)
