"""Shared fixtures for the verification suite; builders live in
case_builders.py so the figure scripts can reuse them without importing a
conftest module."""

import pytest

import interfaces.coolprop_interface as cpi
from case_builders import ACOUSTIC_FLUID, VERIFICATION_FLUID


@pytest.fixture
def constant_fluid():
    """Register the verification fluid for FluidType.CONSTANT and restore the
    previous registration on teardown (no cross-test leakage). Yields the
    ConstantFluidProperties in use."""
    previous = cpi.set_constant_fluid_properties(VERIFICATION_FLUID)
    yield VERIFICATION_FLUID
    cpi.set_constant_fluid_properties(previous)


@pytest.fixture
def acoustic_fluid():
    """Same as constant_fluid but with the low-viscosity wave-dynamics fluid
    used by V3/V4 (see case_builders.ACOUSTIC_FLUID)."""
    previous = cpi.set_constant_fluid_properties(ACOUSTIC_FLUID)
    yield ACOUSTIC_FLUID
    cpi.set_constant_fluid_properties(previous)
