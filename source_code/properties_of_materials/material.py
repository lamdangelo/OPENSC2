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
    density_mgb2,
    isobaric_specific_heat_mgb2,
    thermal_conductivity_mgb2,
)
from properties_of_materials.niobium3_tin import (
    density_nb3sn,
    isobaric_specific_heat_nb3sn,
    thermal_conductivity_nb3sn,
)
from properties_of_materials.niobium_titanium import (
    density_nbti,
    isobaric_specific_heat_nbti,
    thermal_conductivity_nbti,
)
from properties_of_materials.niobium_titanium_w7x import (
    density_nbti_w7x,
    isobaric_specific_heat_nbti_w7x,
    thermal_conductivity_nbti_w7x,
)
from properties_of_materials.rare_earth_123 import (
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
    Material("nb3sn", density_nb3sn, isobaric_specific_heat_nb3sn, thermal_conductivity_nb3sn,
             specific_heat_signature=CP_NB3SN, is_superconductor=True),
    Material("nbti", density_nbti, isobaric_specific_heat_nbti, thermal_conductivity_nbti,
             specific_heat_signature=CP_NBTI, is_superconductor=True),
    Material("nbti-w7x", density_nbti_w7x, isobaric_specific_heat_nbti_w7x, thermal_conductivity_nbti_w7x,
             specific_heat_signature=CP_NBTI_W7X, is_superconductor=True),
    Material("re123", density_re123, isobaric_specific_heat_re123, thermal_conductivity_re123, is_superconductor=True),
    Material("mgb2", density_mgb2, isobaric_specific_heat_mgb2, thermal_conductivity_mgb2,
             conductivity_signature=FIELD, is_superconductor=True),
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
