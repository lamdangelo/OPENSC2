"""Unit tests of the Tier-1 steady-network reference (no main-solver import).

Constant-property, laminar (f = 16/Re) limits with closed forms:
* incompressible single channel: Hagen-Poiseuille mdot = rho A dp D^2/(32 mu L);
* three channels with far-side valves: Kirchhoff solution of the linear
  resistor network R_k = 32 mu L_k / (D^2 rho A);
* weakly compressible channel: p(x) is the exact solution of
  dp/dx = -32 mu v/D^2 with rho = rho0 (1 + kappa (p - p0)), i.e. mdot const.
"""

import numpy as np
import pytest

from reference.steady_network import (
    ChannelSpec,
    channel_mass_flow,
    integrate_channel,
    solve_steady_network,
)

RHO0, MU, CV = 1000.0, 1.0e-3, 4000.0
P0 = 1.0e5


def constant_properties(kappa=0.0):
    def properties(temperature, pressure):
        return dict(
            density=RHO0 * (1.0 + kappa * (pressure - P0)),
            isothermal_compressibility=kappa / (1.0 + kappa * (pressure - P0)),
            isobaric_expansion_coefficient=0.0,
            isochoric_specific_heat=CV,
            viscosity=MU,
        )

    return properties


def laminar(reynolds):
    return 16.0 / reynolds


def channel(identifier, length, diameter=1.0e-2):
    return ChannelSpec(identifier, length, np.pi / 4 * diameter ** 2, diameter, laminar)


def hagen_poiseuille_flow(drop, length, diameter):
    area = np.pi / 4 * diameter ** 2
    return RHO0 * area * drop * diameter ** 2 / (32.0 * MU * length)


def test_incompressible_single_channel_matches_hagen_poiseuille():
    spec = channel("c1", 10.0)
    drop = 200.0
    mass_flow = channel_mass_flow(P0 + drop, P0, 300.0, spec, constant_properties())
    assert mass_flow == pytest.approx(hagen_poiseuille_flow(drop, 10.0, 1.0e-2), rel=1e-12)
    profile = integrate_channel(mass_flow, P0 + drop, 300.0, spec, constant_properties(),
                                profile_points=11)
    # Linear pressure, constant temperature (beta = 0 -> no expansion work,
    # frictional heating raises T linearly: dT/dx = F v / cv).
    assert np.allclose(profile.pressure, P0 + drop * (1 - profile.x / 10.0), rtol=1e-12)
    assert np.all(np.diff(profile.temperature) > 0.0)


def test_three_channels_with_valves_match_kirchhoff():
    lengths = (20.0, 30.0, 40.0)
    diameter = 1.0e-2
    channels = [channel(f"c{k}", L, diameter) for k, L in enumerate(lengths)]
    area = np.pi / 4 * diameter ** 2
    # Hagen-Poiseuille in (p, mdot) variables: R = 32 mu L / (D^2 rho A).
    resistances = np.array([32.0 * MU * L / (diameter ** 2 * RHO0 * area) for L in lengths])
    R_s, R_r = 1.0e4, 2.0e4  # comparable to the channel resistances (~1e5)
    p_s, p_r = P0 + 1000.0, P0
    total_resistance = R_s + R_r + 1.0 / np.sum(1.0 / resistances)
    total = (p_s - p_r) / total_resistance
    drop = total / np.sum(1.0 / resistances)
    expected = drop / resistances

    solution = solve_steady_network(channels, p_s, p_r, 300.0, R_s, R_r,
                                    constant_properties())
    assert solution.mass_flows == pytest.approx(expected, rel=1e-11)
    assert solution.total_mass_flow == pytest.approx(total, rel=1e-11)
    assert solution.pressure_supply_volume == pytest.approx(p_s - R_s * total, rel=1e-12)
    assert solution.pressure_return_volume == pytest.approx(p_r + R_r * total, rel=1e-12)
    assert solution.root_residual < 1e-12


def test_weakly_compressible_channel_conserves_mass_flow():
    """With rho = rho0 (1 + kappa (p - p0)) and laminar friction, rho dp/dx =
    -(32 mu/D^2) mdot/A is constant up to the O(Mach^2) convective term, so
    (p - p0) + kappa (p - p0)^2 / 2 is linear in x; the reference must keep
    mdot constant and reproduce this to the ODE tolerance (Mach^2 ~ 2e-7 at
    the chosen 20 Pa drop, while the quadratic term is 5e-5 of the linear)."""
    kappa = 5.0e-6  # 1/Pa
    spec = channel("c1", 10.0)
    drop = 20.0
    mass_flow = channel_mass_flow(P0 + drop, P0, 300.0, spec, constant_properties(kappa))
    profile = integrate_channel(mass_flow, P0 + drop, 300.0, spec, constant_properties(kappa),
                                profile_points=21)
    # Mass flow reconstructed from the profile is constant to roundoff.
    reconstructed = profile.density * spec.cross_section * profile.velocity
    assert np.allclose(reconstructed, mass_flow, rtol=1e-14)
    excess = profile.pressure - P0
    quadratic = excess + 0.5 * kappa * excess ** 2
    slope = np.diff(quadratic) / np.diff(profile.x)
    assert np.allclose(slope, slope[0], rtol=1e-6)
    # The quadratic term is resolved (the linear-p assumption would fail).
    linear_slope = np.diff(excess) / np.diff(profile.x)
    assert not np.allclose(linear_slope, linear_slope[0], rtol=1e-6)
    assert profile.pressure[-1] == pytest.approx(P0, abs=1e-8)
