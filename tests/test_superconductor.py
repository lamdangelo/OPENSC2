"""Superconductor objects (properties_of_materials.material) call the
critical-surface correlations with exactly the arguments the strand
components used to pass, and the strand dispatch goes through them."""

import numpy as np
import pytest

from factories import make_conductor, make_stack, make_strand_mixed
from opensc2.properties_of_materials import magnesium_diboride as mgb2
from opensc2.properties_of_materials import niobium3_tin as nb3sn
from opensc2.properties_of_materials import niobium_titanium as nbti
from opensc2.properties_of_materials import niobium_titanium_w7x as nbti_w7x
from opensc2.properties_of_materials import rare_earth_123 as re123
from opensc2.properties_of_materials.material import (
    SUPERCONDUCTORS,
    CriticalSurfaceParameters,
    Superconductor,
    UserDefinedScaling,
    material_for,
    superconductor_for,
)

T = np.linspace(4.5, 12.0, 6)
B = np.linspace(1.0, 11.0, 6)
EPS = np.full(6, -0.005)
JOP = np.full(6, 2.0e8)
P = CriticalSurfaceParameters(16.22, 32.35, 1.21508e11)  # CASE_1 Nb3Sn-like scaling
TC0, BC20, C0 = P.critical_temperature_at_zero_field, P.upper_critical_field_at_zero_temperature, P.scaling_constant
# MgB2 needs its own (physical) critical surface: Tc0 39 K, Bc20 14 T, a
# small operating current density (its Tcs inversion bisects Jc(T) = Jop).
P_MGB2 = CriticalSurfaceParameters(39.0, 14.0, 1.0e9)
TC0_MGB2, BC20_MGB2, C0_MGB2 = 39.0, 14.0, 1.0e9
JOP_MGB2 = np.full(6, 1.0e6)
PARAMETERS = {"mgb2": (P_MGB2, JOP_MGB2)}


def test_registry():
    assert set(SUPERCONDUCTORS) == {"nb3sn", "nbti", "nbti-w7x", "re123", "mgb2"}
    assert superconductor_for("ybco") is material_for("re123")
    assert isinstance(material_for("nbti"), Superconductor)
    assert superconductor_for("nb3sn").needs_strain and not superconductor_for("nbti").needs_strain
    assert isinstance(superconductor_for("scaling.dat"), UserDefinedScaling)
    with pytest.raises(KeyError):
        superconductor_for("cu")
    with pytest.raises(KeyError):
        superconductor_for("unobtainium")


@pytest.mark.parametrize(
    "name, jc, tc, tcs",
    [
        ("nbti", lambda: nbti.critical_current_density_nbti(T, B, BC20, C0, TC0),
         lambda: nbti.critical_temperature_nbti(B, BC20, TC0),
         lambda: nbti.current_sharing_temperature_nbti(B, JOP, BC20, C0, TC0)),
        ("nbti-w7x", lambda: nbti_w7x.critical_current_density_nbti_w7x(T, B, BC20, C0, TC0),
         lambda: nbti_w7x.critical_temperature_nbti_w7x(B, BC20, TC0),
         lambda: nbti_w7x.current_sharing_temperature_nbti_w7x(B, JOP, BC20, C0, TC0)),
        ("nb3sn", lambda: nb3sn.critical_current_density_nb3sn(T, B, EPS, TC0, BC20, C0),
         lambda: nb3sn.critical_temperature_nb3sn(B, EPS, TC0, BC20),
         lambda: nb3sn.current_sharing_temperature_nb3sn(B, EPS, JOP, TC0, BC20, C0)),
        ("re123", lambda: re123.critical_current_density_re123(T, B, TC0, BC20, C0),
         lambda: TC0 * np.ones(T.shape),
         lambda: re123.current_sharing_temperature_re123(B, JOP, TC0, BC20, C0)),
        ("mgb2", lambda: mgb2.critical_current_density_mgb2(T, B, BC20_MGB2, C0_MGB2, TC0_MGB2),
         lambda: TC0_MGB2 * np.ones(T.shape),
         lambda: mgb2.current_sharing_temperature_mgb2(B.copy(), JOP_MGB2, BC20_MGB2, C0_MGB2, TC0_MGB2)),
    ],
)
def test_critical_surface_calls_match_the_correlations_bitwise(name, jc, tc, tcs):
    superconductor = superconductor_for(name)
    parameters, jop = PARAMETERS.get(name, (P, JOP))
    strain = EPS if superconductor.needs_strain else None
    np.testing.assert_array_equal(superconductor.critical_current_density(T, B, parameters, strain=strain), jc())
    np.testing.assert_array_equal(superconductor.critical_temperature(B, parameters, strain=strain), tc())
    np.testing.assert_array_equal(
        superconductor.current_sharing_temperature(B.copy(), jop, parameters, strain=strain), tcs()
    )


@pytest.mark.parametrize("name", ["nbti", "mgb2", "nbti-w7x"])
def test_strand_critical_properties_and_tcs_go_through_the_superconductor(name):
    parameters, jop = PARAMETERS.get(name, (P, JOP))
    strand = make_strand_mixed(
        superconducting_material=name,
        cross_section=6.2e-5,
        critical_current_definition_mode=1,
        critical_temperature_at_0T=parameters.critical_temperature_at_zero_field,
        upper_critical_field_at_0K=parameters.upper_critical_field_at_zero_temperature,
        critical_current_scaling_constant=parameters.scaling_constant,
    )
    fields = strand.node_fields
    fields.temperature = T.copy()
    fields.B_field = B.copy()
    fields.op_current = jop * strand.cross_section["sc"]
    strand.eval_critical_properties(fields)
    strand.eval_tcs(fields)
    sc = superconductor_for(name)
    np.testing.assert_array_equal(fields.J_critical, sc.critical_current_density(T, B, strand.critical_surface))
    np.testing.assert_array_equal(fields.T_critical, sc.critical_temperature(B, strand.critical_surface))
    jop = np.abs(fields.op_current) / strand.cross_section["sc"]
    np.testing.assert_array_equal(fields.T_cur_sharing, sc.current_sharing_temperature(B, jop, strand.critical_surface))
    np.testing.assert_array_equal(fields.T_cur_sharing_min, fields.T_cur_sharing)


def test_nb3sn_strand_uses_its_strain_field():
    strand = make_strand_mixed(critical_current_definition_mode=1)  # CASE_1 Nb3Sn defaults
    strand.operations.strain_mode, strand.operations.strain_value = 1, -0.0068
    strand.get_eps(make_conductor(number_of_nodes=6))
    fields = strand.node_fields
    fields.temperature = T.copy()
    fields.B_field = B.copy()
    strand.eval_critical_properties(fields)
    eps = np.full(6, -0.0068)
    p = strand.critical_surface
    np.testing.assert_array_equal(
        fields.J_critical,
        nb3sn.critical_current_density_nb3sn(T, B, eps, p.critical_temperature_at_zero_field,
                                              p.upper_critical_field_at_zero_temperature, p.scaling_constant),
    )


def test_stack_resolves_its_superconductor_by_alias():
    stack = make_stack()  # superconducting_material "ybco"
    assert stack.superconductor is material_for("re123")
