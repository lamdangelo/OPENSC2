"""Material objects with a uniform property interface.

The property correlations in this package are plain functions with
material-specific signatures (``thermal_conductivity_cu_nist(T, B, RRR)``
versus ``thermal_conductivity_ss(T)``, four different isobaric specific heat
signatures for the superconductors). The components used to hold arrays of
those functions and branch on ``func.__name__`` to know how to call them.

A :class:`Material` hides the signature: every property takes the
temperature plus optional keyword arguments (field, RRR, critical
temperatures) and each material picks what its correlation needs, calling
the very same function with the very same arguments as before. The
:data:`MATERIALS` registry maps the input-file names (and aliases such as
``re123``/``ybco``) to the material objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from properties_of_materials.aluminium import (
    density_al,
    electrical_resistivity_al,
    isobaric_specific_heat_al,
    thermal_conductivity_al,
)
from properties_of_materials.aluminium_6063 import (
    density_al6063,
    electrical_resistivity_al6063,
    isobaric_specific_heat_al6063,
    thermal_conductivity_al6063,
)
from properties_of_materials.copper import (
    density_cu,
    electrical_resistivity_cu_nist,
    isobaric_specific_heat_cu_nist,
    thermal_conductivity_cu_nist,
)
from properties_of_materials.glass_epoxy import (
    density_ge,
    electrical_resistivity_ge,
    isobaric_specific_heat_ge,
    thermal_conductivity_ge,
)
from properties_of_materials.hastelloy_c276 import (
    density_hc276,
    electrical_resistivity_hc276,
    isobaric_specific_heat_hc276,
    thermal_conductivity_hc276,
)
from properties_of_materials.magnesium_diboride import (
    critical_current_density_mgb2,
    current_sharing_temperature_mgb2,
    density_mgb2,
    isobaric_specific_heat_mgb2,
    thermal_conductivity_mgb2,
)
from properties_of_materials.niobium3_tin import (
    critical_current_density_nb3sn,
    critical_temperature_nb3sn,
    current_sharing_temperature_nb3sn,
    density_nb3sn,
    isobaric_specific_heat_nb3sn,
    thermal_conductivity_nb3sn,
)
from properties_of_materials.niobium_titanium import (
    critical_current_density_nbti,
    critical_temperature_nbti,
    current_sharing_temperature_nbti,
    density_nbti,
    isobaric_specific_heat_nbti,
    thermal_conductivity_nbti,
)
from properties_of_materials.niobium_titanium_w7x import (
    critical_current_density_nbti_w7x,
    critical_temperature_nbti_w7x,
    current_sharing_temperature_nbti_w7x,
    density_nbti_w7x,
    isobaric_specific_heat_nbti_w7x,
    thermal_conductivity_nbti_w7x,
)
from properties_of_materials.rare_earth_123 import (
    critical_current_density_re123,
    current_sharing_temperature_re123,
    density_re123,
    isobaric_specific_heat_re123,
    thermal_conductivity_re123,
)
from properties_of_materials.silver import (
    density_ag,
    electrical_resistivity_ag,
    isobaric_specific_heat_ag,
    thermal_conductivity_ag,
)
from properties_of_materials.solder_sn60_pb40 import (
    density_sn60pb40,
    electrical_resistivity_sn60pb40,
    isobaric_specific_heat_sn60pb40,
    thermal_conductivity_sn60pb40,
)
from properties_of_materials.stainless_steel import (
    density_ss,
    electrical_resistivity_ss,
    isobaric_specific_heat_ss,
    thermal_conductivity_ss,
)

# How a correlation is called (which keyword arguments it consumes).
TEMPERATURE_ONLY = "T"
FIELD_AND_RRR = "T,B,RRR"  # copper: k and rho_el
FIELD = "T,B"  # MgB2 thermal conductivity
CP_NB3SN = "T,Tcs,Tc,Tc0"
CP_NBTI = "T,B,Tcs,Tc"
CP_NBTI_W7X = "T,Tc,Tc0"

# How the critical-surface correlations of a superconductor are called
# (argument order of Jc, Tc and Tcs in the per-material modules).
CRITICAL_B_BC20_C0_TC0 = "B,Bc20,c0,Tc0"  # NbTi, NbTi-W7X, MgB2
CRITICAL_TC0_BC20_C0 = "Tc0,Bc20,c0"  # RE123
CRITICAL_STRAIN = "eps,Tc0,Bc20,c0"  # Nb3Sn (strain-dependent scaling)


@dataclass(frozen=True)
class Material:
    """One solid material: four property correlations and how to call them."""

    name: str
    _density: Callable
    _isobaric_specific_heat: Callable
    _thermal_conductivity: Callable
    _electrical_resistivity: Optional[Callable] = None
    specific_heat_signature: str = TEMPERATURE_ONLY
    conductivity_signature: str = TEMPERATURE_ONLY
    resistivity_signature: str = TEMPERATURE_ONLY
    is_superconductor: bool = False

    def density(self, temperature) -> np.ndarray:
        return self._density(temperature)

    def isobaric_specific_heat(
        self,
        temperature,
        *,
        magnetic_field=None,
        current_sharing_temperature=None,
        critical_temperature=None,
        critical_temperature_at_zero_field=None,
    ) -> np.ndarray:
        if self.specific_heat_signature == CP_NB3SN:
            return self._isobaric_specific_heat(
                temperature, current_sharing_temperature, critical_temperature,
                critical_temperature_at_zero_field,
            )
        if self.specific_heat_signature == CP_NBTI:
            return self._isobaric_specific_heat(
                temperature, magnetic_field, current_sharing_temperature,
                critical_temperature,
            )
        if self.specific_heat_signature == CP_NBTI_W7X:
            return self._isobaric_specific_heat(
                temperature, critical_temperature, critical_temperature_at_zero_field
            )
        return self._isobaric_specific_heat(temperature)

    def thermal_conductivity(
        self, temperature, *, magnetic_field=None, residual_resistivity_ratio=None
    ) -> np.ndarray:
        if self.conductivity_signature == FIELD_AND_RRR:
            return self._thermal_conductivity(
                temperature, magnetic_field, residual_resistivity_ratio
            )
        if self.conductivity_signature == FIELD:
            return self._thermal_conductivity(temperature, magnetic_field)
        return self._thermal_conductivity(temperature)

    def electrical_resistivity(
        self, temperature, *, magnetic_field=None, residual_resistivity_ratio=None
    ) -> np.ndarray:
        if self._electrical_resistivity is None:
            raise NotImplementedError(
                f"no electrical resistivity correlation for material {self.name!r}"
            )
        if self.resistivity_signature == FIELD_AND_RRR:
            return self._electrical_resistivity(
                temperature, magnetic_field, residual_resistivity_ratio
            )
        return self._electrical_resistivity(temperature)


@dataclass(frozen=True)
class CriticalSurfaceParameters:
    """Per-strand parameters of the critical-surface scaling."""

    critical_temperature_at_zero_field: float  # Tc0 [K]
    upper_critical_field_at_zero_temperature: float  # Bc20 [T]
    scaling_constant: float  # c0 (physical definition)


@dataclass(frozen=True)
class Superconductor(Material):
    """A superconducting material: thermal properties plus the critical
    surface (Jc, Tc, Tcs), called with the per-material argument order."""

    _critical_current_density: Optional[Callable] = None
    _critical_temperature: Optional[Callable] = None  # None: Tc = Tc0
    _current_sharing_temperature: Optional[Callable] = None
    critical_signature: str = CRITICAL_B_BC20_C0_TC0

    @property
    def needs_strain(self) -> bool:
        """True when the critical surface depends on the strain field."""
        return self.critical_signature == CRITICAL_STRAIN

    def critical_current_density(
        self, temperature, magnetic_field, parameters: CriticalSurfaceParameters, *, strain=None
    ) -> np.ndarray:
        tc0 = parameters.critical_temperature_at_zero_field
        bc20 = parameters.upper_critical_field_at_zero_temperature
        c0 = parameters.scaling_constant
        if self.critical_signature == CRITICAL_STRAIN:
            return self._critical_current_density(temperature, magnetic_field, strain, tc0, bc20, c0)
        if self.critical_signature == CRITICAL_TC0_BC20_C0:
            return self._critical_current_density(temperature, magnetic_field, tc0, bc20, c0)
        return self._critical_current_density(temperature, magnetic_field, bc20, c0, tc0)

    def critical_temperature(
        self, magnetic_field, parameters: CriticalSurfaceParameters, *, strain=None
    ) -> np.ndarray:
        tc0 = parameters.critical_temperature_at_zero_field
        bc20 = parameters.upper_critical_field_at_zero_temperature
        if self._critical_temperature is None:
            # No Tc(B) correlation: the zero-field value everywhere.
            return tc0 * np.ones(np.shape(magnetic_field))
        if self.critical_signature == CRITICAL_STRAIN:
            return self._critical_temperature(magnetic_field, strain, tc0, bc20)
        return self._critical_temperature(magnetic_field, bc20, tc0)

    def current_sharing_temperature(
        self, magnetic_field, operating_current_density, parameters: CriticalSurfaceParameters, *, strain=None
    ) -> np.ndarray:
        tc0 = parameters.critical_temperature_at_zero_field
        bc20 = parameters.upper_critical_field_at_zero_temperature
        c0 = parameters.scaling_constant
        if self.critical_signature == CRITICAL_STRAIN:
            return self._current_sharing_temperature(
                magnetic_field, strain, operating_current_density, tc0, bc20, c0
            )
        if self.critical_signature == CRITICAL_TC0_BC20_C0:
            return self._current_sharing_temperature(
                magnetic_field, operating_current_density, tc0, bc20, c0
            )
        return self._current_sharing_temperature(
            magnetic_field, operating_current_density, bc20, c0, tc0
        )


class UserDefinedScaling:
    """Placeholder for the legacy ``scaling.dat`` option: the critical surface
    is meant to come from the user scaling file read by
    StrandComponent.user_scaling_margin; no correlation is evaluated."""

    name = USER_SCALING_NAME = "scaling.dat"
    needs_strain = False

    def critical_current_density(self, *args, **kwargs):
        raise NotImplementedError(
            "the user-defined scaling (scaling.dat) provides no critical current density correlation"
        )


_MATERIALS = [
    Material("al", density_al, isobaric_specific_heat_al, thermal_conductivity_al, electrical_resistivity_al),
    Material("al6063", density_al6063, isobaric_specific_heat_al6063, thermal_conductivity_al6063, electrical_resistivity_al6063),
    Material("cu", density_cu, isobaric_specific_heat_cu_nist, thermal_conductivity_cu_nist, electrical_resistivity_cu_nist,
             conductivity_signature=FIELD_AND_RRR, resistivity_signature=FIELD_AND_RRR),
    Material("ag", density_ag, isobaric_specific_heat_ag, thermal_conductivity_ag, electrical_resistivity_ag),
    Material("ge", density_ge, isobaric_specific_heat_ge, thermal_conductivity_ge, electrical_resistivity_ge),
    Material("hc276", density_hc276, isobaric_specific_heat_hc276, thermal_conductivity_hc276, electrical_resistivity_hc276),
    Material("sn60pb40", density_sn60pb40, isobaric_specific_heat_sn60pb40, thermal_conductivity_sn60pb40, electrical_resistivity_sn60pb40),
    Material("ss", density_ss, isobaric_specific_heat_ss, thermal_conductivity_ss, electrical_resistivity_ss),
    Superconductor("nb3sn", density_nb3sn, isobaric_specific_heat_nb3sn, thermal_conductivity_nb3sn,
                   specific_heat_signature=CP_NB3SN, is_superconductor=True,
                   _critical_current_density=critical_current_density_nb3sn,
                   _critical_temperature=critical_temperature_nb3sn,
                   _current_sharing_temperature=current_sharing_temperature_nb3sn,
                   critical_signature=CRITICAL_STRAIN),
    Superconductor("nbti", density_nbti, isobaric_specific_heat_nbti, thermal_conductivity_nbti,
                   specific_heat_signature=CP_NBTI, is_superconductor=True,
                   _critical_current_density=critical_current_density_nbti,
                   _critical_temperature=critical_temperature_nbti,
                   _current_sharing_temperature=current_sharing_temperature_nbti),
    Superconductor("nbti-w7x", density_nbti_w7x, isobaric_specific_heat_nbti_w7x, thermal_conductivity_nbti_w7x,
                   specific_heat_signature=CP_NBTI_W7X, is_superconductor=True,
                   _critical_current_density=critical_current_density_nbti_w7x,
                   _critical_temperature=critical_temperature_nbti_w7x,
                   _current_sharing_temperature=current_sharing_temperature_nbti_w7x),
    Superconductor("re123", density_re123, isobaric_specific_heat_re123, thermal_conductivity_re123,
                   is_superconductor=True,
                   _critical_current_density=critical_current_density_re123,
                   _current_sharing_temperature=current_sharing_temperature_re123,
                   critical_signature=CRITICAL_TC0_BC20_C0),
    Superconductor("mgb2", density_mgb2, isobaric_specific_heat_mgb2, thermal_conductivity_mgb2,
                   conductivity_signature=FIELD, is_superconductor=True,
                   _critical_current_density=critical_current_density_mgb2,
                   _current_sharing_temperature=current_sharing_temperature_mgb2),
]

MATERIALS = {material.name: material for material in _MATERIALS}
# Input-file aliases (case-insensitive) of the canonical names.
ALIASES = {
    "ybco": "re123",
    "rebco": "re123",
    "copper": "cu",
    "aluminium": "al",
    "aluminum": "al",
    "aluminium_6063": "al6063",
    "aluminum_6063": "al6063",
    "stainless_steel": "ss",
}


def material_for(name: str) -> Material:
    """The material registered under ``name`` (canonical or alias)."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}") from None


def superconductor_for(name: str):
    """The :class:`Superconductor` registered under ``name`` (canonical or
    alias), or the :class:`UserDefinedScaling` placeholder for the legacy
    ``scaling.dat`` option.

    Raises:
        KeyError: for an unknown name or a material that is not a superconductor.
    """
    if str(name).strip().lower() == UserDefinedScaling.name:
        return UserDefinedScaling()
    material = material_for(name)
    if not isinstance(material, Superconductor):
        raise KeyError(f"{name!r} is not a superconductor; known: {sorted(SUPERCONDUCTORS)}")
    return material


SUPERCONDUCTORS = {name: material for name, material in MATERIALS.items() if isinstance(material, Superconductor)}
