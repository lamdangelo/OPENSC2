"""Fluid property port.

The coolant, the fluid component and the lumped hydraulic network need
thermophysical properties of the working fluid as functions of (T, p). They
depend on this Protocol only; the CoolProp-backed implementation (with its
tabulated splines) and the constant-property verification fluid live in
``adapters.fluid_properties``. The composition root (simulation.py, or the
test configuration) registers a factory with :func:`register_provider_factory`;
domain objects resolve their provider through :func:`provider_for` unless one
is injected into their constructor.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import numpy as np

from hydraulics.hydraulic_flags import FluidType

# Property names used by the channel coolant, mapped to the CoolProp output
# aliases understood by every provider (``properties`` takes such a mapping).
# Formerly Simulation.fluid_prop_aliases.
COOLANT_PROPERTY_ALIASES: Dict[str, str] = dict(
    isobaric_expansion_coefficient="isobaric_expansion_coefficient",
    isothermal_compressibility="isothermal_compressibility",  # 1/Pa
    Prandtl="Prandtl",
    total_density="Dmass",
    total_dynamic_viscosity="viscosity",
    total_enthalpy="Hmass",
    total_isobaric_specific_heat="Cpmass",
    total_isochoric_specific_heat="Cvmass",
    total_speed_of_sound="speed_of_sound",
    total_thermal_conductivity="conductivity",
)


class FluidPropertyProvider(Protocol):
    """Thermophysical properties of one fluid as functions of (T, p)."""

    fluid_type: FluidType

    def mass_density(self, temperature, pressure) -> np.ndarray: ...

    def viscosity(self, temperature, pressure) -> np.ndarray: ...

    def properties(
        self, aliases: Dict[str, str], temperature, pressure
    ) -> Dict[str, np.ndarray]:
        """{name: values} for every (name, alias) in ``aliases``."""
        ...


ProviderFactory = Callable[[FluidType], FluidPropertyProvider]

_provider_factory: Optional[ProviderFactory] = None


def register_provider_factory(factory: Optional[ProviderFactory]) -> Optional[ProviderFactory]:
    """Install the factory that :func:`provider_for` uses (composition root);
    returns the previous one so callers can restore it."""
    global _provider_factory
    previous = _provider_factory
    _provider_factory = factory
    return previous


def provider_for(fluid_type: FluidType) -> FluidPropertyProvider:
    if _provider_factory is None:
        raise RuntimeError(
            "No fluid property provider registered. The composition root must "
            "call ports.fluid_properties.register_provider_factory(...) (the "
            "Simulation facade registers adapters.fluid_properties.provider_for)."
        )
    return _provider_factory(fluid_type)
