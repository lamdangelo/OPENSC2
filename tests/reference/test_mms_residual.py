"""Unit tests of the Tier-4 MMS residual (no main-solver import).

With constant properties (beta = 0, phi = 0, gamma = 1, c^2 = 1/(rho kappa_T))
and no friction the fluid system is the linear acoustic system; a travelling
wave p = p0 + eps cos(k x - omega t), v = eps cos(k x - omega t)/(rho c),
T = const is an exact solution, so the residual of both formulations must
vanish to finite-difference accuracy. The finite-difference error itself is
checked to shrink with the relative step (second order).
"""

import numpy as np
import pytest

from reference.mms_residual import ManufacturedSolution, residual

RHO, C, CV = 1000.0, 30.0, 4000.0
KAPPA = 1.0 / (RHO * C ** 2)
AREA, DIAMETER = 1.0e-4, 1.0e-2


def constant_properties(temperature, pressure):
    shape = np.broadcast(temperature, pressure).shape
    ones = np.ones(shape)
    return dict(
        density=RHO * ones,
        isothermal_compressibility=KAPPA * ones,
        isobaric_expansion_coefficient=0.0 * ones,
        isochoric_specific_heat=CV * ones,
        isobaric_specific_heat=CV * ones,
        speed_of_sound=C * ones,
        viscosity=1.0e-3 * ones,
    )


def no_friction(reynolds):
    return np.zeros_like(np.asarray(reynolds, dtype=float))


class TravellingWave(ManufacturedSolution):
    """Exact acoustic solution dressed as a ManufacturedSolution."""

    eps = 1.0e-3  # Pa: the O(eps^2) convective terms are then 1e-9 relative
    wavenumber = 2 * np.pi / 5.0

    def pressure(self, x, t):
        return self.pressure_0 + self.eps * np.cos(self.wavenumber * (x - C * t))

    def mass_flow(self, x, t):
        velocity = self.eps * np.cos(self.wavenumber * (x - C * t)) / (RHO * C)
        # Exact mass flow of the linear solution: rho0 A v (density
        # perturbation times velocity is second order in eps).
        return RHO * AREA * velocity

    def temperature(self, x, t):
        return self.temperature_0 + 0.0 * np.asarray(x)


@pytest.mark.parametrize("formulation", ["velocity", "mass_flow"])
def test_exact_acoustic_solution_has_small_residual(formulation):
    # A small base pressure keeps the finite-difference cancellation error
    # (p0 * eps_machine / step) far below the eps-scaled residual terms.
    wave = TravellingWave(mass_flow_0=1.0, pressure_0=1.0e3, pressure_drop=0.0,
                          length=10.0, period=1.0)
    x = np.linspace(0.5, 9.5, 37)
    rows = residual(wave, x, 0.123, formulation, constant_properties, no_friction,
                    AREA, DIAMETER)
    # Scales of the individual terms: omega eps for the pressure row. The
    # residual is limited by the finite-difference cancellation of the base
    # pressure (p0 eps_machine / (2 h) ~ 1e-6 relative here); the O(eps^2)
    # convective terms are ~1e-9 relative and the truncation ~1e-11.
    omega = wave.wavenumber * C
    scale_p = omega * wave.eps
    assert np.abs(rows["pressure"]).max() < 1e-5 * scale_p
    scale_v = omega * wave.eps / (RHO * C)
    if formulation == "mass_flow":
        scale_v *= RHO * AREA
    assert np.abs(rows["velocity"]).max() < 1e-5 * scale_v
    assert np.abs(rows["temperature"]).max() == 0.0


def test_finite_difference_step_convergence():
    """The residual of the generic manufactured solution converges as the
    relative step shrinks (central differences: second order)."""
    solution = ManufacturedSolution(mass_flow_0=2.0e-3, pressure_0=5.0e5,
                                    pressure_drop=5.0e4, length=20.0, period=0.5)
    x = np.linspace(1.0, 19.0, 19)
    fine = residual(solution, x, 0.137, "mass_flow", constant_properties, no_friction,
                    AREA, DIAMETER, relative_step=1e-7)
    errors = []
    for step in (1e-3, 1e-4):
        rows = residual(solution, x, 0.137, "mass_flow", constant_properties, no_friction,
                        AREA, DIAMETER, relative_step=step)
        errors.append(np.abs(rows["pressure"] - fine["pressure"]).max())
    assert errors[1] < 1e-1 * errors[0]


def test_manufactured_boundary_values_are_constant_in_time():
    solution = ManufacturedSolution(mass_flow_0=2.0e-3, pressure_0=5.0e5,
                                    pressure_drop=5.0e4, length=20.0, period=0.5)
    times = np.linspace(0.0, 0.5, 7)
    assert np.allclose(solution.mass_flow(0.0, times), solution.mass_flow_0)
    assert np.allclose(solution.pressure(20.0, times), solution.pressure_0 - solution.pressure_drop)
    assert np.allclose(solution.temperature(0.0, times), solution.temperature_0)


def test_velocity_manufactured_solution_is_consistent():
    """Tier-5 variant: v is prescribed, mdot = rho A v derived; the inlet
    mass flow and the boundary values stay those of the Tier-4 solution."""
    from reference.mms_residual import VelocityManufacturedSolution

    solution = VelocityManufacturedSolution(mass_flow_0=2.0e-3, pressure_0=5.0e5,
                                            pressure_drop=5.0e4, length=20.0, period=0.5,
                                            properties=constant_properties, area=AREA)
    times = np.linspace(0.0, 0.5, 7)
    x = np.linspace(0.0, 20.0, 41)
    assert np.isclose(solution.velocity_0, 2.0e-3 / (RHO * AREA))
    assert np.allclose(solution.mass_flow(0.0, times), solution.mass_flow_0)
    assert np.allclose(solution.velocity(0.0, times), solution.velocity_0)
    # mdot / (rho A) round-trips to the prescribed velocity at every point.
    assert np.allclose(solution.mass_flow(x, 0.137) / (RHO * AREA), solution.velocity(x, 0.137))
    # The prescribed velocity has the sinusoidal shape (amplitude 0.2 at t = 0).
    assert np.isclose(solution.velocity(5.0, 0.0), 1.2 * solution.velocity_0)
    assert np.isclose(solution.velocity(15.0, 0.0), 0.8 * solution.velocity_0)
