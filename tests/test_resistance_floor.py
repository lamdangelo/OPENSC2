"""Unit tests of the superconducting resistance floor of the steady
current-consistency solve (conductor operation ``electric_resistance_floor``).

Below Ic the power law gives R -> 0, so the split of the transport current
among parallel superconducting strands is undetermined and the Newton solve
drifts (seen on the HELIAS-VIPER 2D runs: loop currents of hundreds of kA
between stacks of a cold double layer, then a one-step thermal explosion).
The floor pins R >= floor * E0 * L_e / Ic for the superconducting elements.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from components.solid.strand_component import StrandComponent


def make_strand(number_of_elements=5, floor=1.0e-3, consistency=True):
    strand = StrandComponent.__new__(StrandComponent)
    strand.identifier = "STACK1"
    strand.inputs = SimpleNamespace(flux_flow_electric_field=1.0e-4, power_law_exponent=20)
    strand.gauss_fields = SimpleNamespace(electric_resistance=np.zeros(number_of_elements))
    lengths = np.full(number_of_elements, 0.5)
    conductor = SimpleNamespace(
        operations=SimpleNamespace(electric_resistance_floor=floor, electric_current_consistency=consistency),
        mesh=SimpleNamespace(number_of_elements=number_of_elements),
        node_distance=pd.DataFrame({("StrandComponent", "STACK1"): lengths}),
    )
    return strand, conductor


def test_floor_raises_only_the_superconducting_elements_below_it():
    strand, conductor = make_strand()
    critical_current = np.full(5, 8.0e4)
    strand.gauss_fields.electric_resistance[:] = [0.0, 1e-20, 1e-16, 1e-12, 1e-9]
    indices = np.array([0, 1, 2, 3])  # element 4 is not superconducting
    strand.apply_superconducting_resistance_floor(conductor, indices, critical_current[indices])
    expected_floor = 1.0e-3 * 1.0e-4 * 0.5 / 8.0e4  # 6.25e-13 Ohm
    np.testing.assert_allclose(strand._resistance_floor_gauss[:4], expected_floor)
    assert strand._resistance_floor_gauss[4] == 0.0
    resistance = strand.gauss_fields.electric_resistance
    np.testing.assert_allclose(resistance[:3], expected_floor)
    assert resistance[3] == 1e-12  # above the floor: untouched
    assert resistance[4] == 1e-9  # not in the superconducting set: untouched


@pytest.mark.parametrize("floor, consistency", [(0.0, True), (1e-3, False), (None, True)])
def test_floor_is_a_no_op_when_disabled(floor, consistency):
    strand, conductor = make_strand(floor=floor, consistency=consistency)
    strand.gauss_fields.electric_resistance[:] = 1e-20
    strand.apply_superconducting_resistance_floor(conductor, np.arange(5), np.full(5, 8.0e4))
    assert strand._resistance_floor_gauss is None
    np.testing.assert_array_equal(strand.gauss_fields.electric_resistance, 1e-20)


def test_differential_resistance_uses_the_floor_where_it_binds():
    strand, conductor = make_strand()
    critical_current = np.full(5, 8.0e4)
    strand.gauss_fields.electric_resistance[:] = [1e-20, 1e-20, 1e-11, 1e-11, 1e-9]
    sc = np.array([0, 1, 2, 3])
    strand.apply_superconducting_resistance_floor(conductor, sc, critical_current[sc])
    strand._electric_resistance_strand_only = strand.gauss_fields.electric_resistance.copy()
    strand._electric_regime_gauss = {
        "sc": sc, "sharing": np.empty(0, dtype=int), "normal": np.array([4]),
        "i_sc": np.zeros(5), "critical_current": critical_current,
    }
    derivative = strand.get_electric_resistance_derivative(conductor)
    floor = strand._resistance_floor_gauss
    np.testing.assert_allclose(derivative[:2], floor[:2])  # floored: d = R_floor
    np.testing.assert_allclose(derivative[2:4], 20 * 1e-11)  # power law: d = n R
    assert derivative[4] == 1e-9  # normal regime: d = R
