"""The Material registry calls the per-material correlations with exactly the
arguments the components used to pass (Stage 8 of the architecture plan)."""

import numpy as np
import pytest

from properties_of_materials.copper import (
    electrical_resistivity_cu_nist,
    isobaric_specific_heat_cu_nist,
    thermal_conductivity_cu_nist,
)
from properties_of_materials.electrical_conductivity import electrical_resistivity_of
from properties_of_materials.magnesium_diboride import thermal_conductivity_mgb2
from properties_of_materials.material import MATERIALS, material_for
from properties_of_materials.niobium3_tin import isobaric_specific_heat_nb3sn
from properties_of_materials.niobium_titanium import isobaric_specific_heat_nbti
from properties_of_materials.niobium_titanium_w7x import isobaric_specific_heat_nbti_w7x
from properties_of_materials.stainless_steel import thermal_conductivity_ss

T = np.linspace(4.5, 60.0, 7)
B = np.linspace(1.0, 6.0, 7)
RRR = 100.0
T_CS = T + 1.0
T_C = T + 3.0
T_C0 = 18.0


def test_registry_names_and_aliases():
    assert {"al", "cu", "nb3sn", "nbti", "nbti-w7x", "re123", "mgb2", "ss", "ge", "al6063"} <= set(MATERIALS)
    assert material_for("ybco") is material_for("re123")
    assert material_for("Cu") is material_for("cu")
    with pytest.raises(KeyError):
        material_for("unobtainium")


def test_signatures_match_direct_calls():
    cu = material_for("cu")
    np.testing.assert_array_equal(
        cu.thermal_conductivity(T, magnetic_field=B, residual_resistivity_ratio=RRR),
        thermal_conductivity_cu_nist(T, B, RRR),
    )
    np.testing.assert_array_equal(
        cu.electrical_resistivity(T, magnetic_field=B, residual_resistivity_ratio=RRR),
        electrical_resistivity_cu_nist(T, B, RRR),
    )
    np.testing.assert_array_equal(cu.isobaric_specific_heat(T, magnetic_field=B), isobaric_specific_heat_cu_nist(T))
    np.testing.assert_array_equal(material_for("ss").thermal_conductivity(T, magnetic_field=B), thermal_conductivity_ss(T))
    np.testing.assert_array_equal(material_for("mgb2").thermal_conductivity(T, magnetic_field=B), thermal_conductivity_mgb2(T, B))
    kwargs = dict(magnetic_field=B, current_sharing_temperature=T_CS, critical_temperature=T_C, critical_temperature_at_zero_field=T_C0)
    np.testing.assert_array_equal(material_for("nb3sn").isobaric_specific_heat(T, **kwargs), isobaric_specific_heat_nb3sn(T, T_CS, T_C, T_C0))
    np.testing.assert_array_equal(material_for("nbti").isobaric_specific_heat(T, **kwargs), isobaric_specific_heat_nbti(T, B, T_CS, T_C))
    np.testing.assert_array_equal(material_for("nbti-w7x").isobaric_specific_heat(T, **kwargs), isobaric_specific_heat_nbti_w7x(T, T_C, T_C0))


def test_superconductor_entries_are_superconductor_objects():
    from properties_of_materials.material import Superconductor

    for name in ("nb3sn", "nbti", "nbti-w7x", "re123", "mgb2"):
        assert isinstance(material_for(name), Superconductor)
    assert not isinstance(material_for("cu"), Superconductor)


def test_superconductors_have_no_normal_resistivity():
    with pytest.raises(NotImplementedError):
        material_for("nbti").electrical_resistivity(T)


def test_alias_dispatcher_goes_through_registry():
    np.testing.assert_array_equal(
        electrical_resistivity_of("copper", T, B, RRR), electrical_resistivity_cu_nist(T, B, RRR)
    )
    with pytest.raises(ValueError):
        electrical_resistivity_of("cu", T)
    with pytest.raises(KeyError):
        electrical_resistivity_of("unobtainium", T)
