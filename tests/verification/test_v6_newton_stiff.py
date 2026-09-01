"""V6-stiff — Newton current consistency far above the critical current.

Same two-strand geometry as V6 (REBCO stack + external copper with
equipotential terminations) driven at I0 = 2 Ic. At this overload the
old Picard fixed-point iteration oscillated (power-law gain ~ n); the
damped Newton solver must converge in a handful of iterations to the
same analytic parallel-circuit fixed point.

Also hosts the Jacobian finite-difference check: the per-element
differential resistance d(V)/d(I) assembled for the Newton matrix is
compared against a central difference of V(I) = R(I)*I in both the
superconducting and the current-sharing regimes.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source_code"))

import analytical  # noqa: E402
from test_v6_current_sharing import (  # noqa: E402
    ELECTRIC_FIELD_CRITERION,
    EXTERNAL_COPPER_AREA,
    POWER_LAW_EXPONENT,
    STACK_STABILIZER_AREA,
    critical_current,
    write_run_directory,
)

STIFF_CURRENT_OVER_IC = 2.0


class StiffCapture:
    """Step callback: mid-length currents, Newton iteration count, and the
    finite-difference Jacobian check (run once, on the first step)."""

    def __init__(self):
        self.stack_current = None
        self.external_current = None
        self.critical_current = None
        self.stack_stabilizer_resistivity = None
        self.external_resistivity = None
        self.newton_iterations = 0
        self.jacobian_check = None

    def __call__(self, simulation) -> bool:
        conductor = simulation.list_of_Conductors[0]
        self.newton_iterations = max(
            self.newton_iterations,
            getattr(conductor, "_steady_newton_iterations", 0),
        )
        stack = None
        for strand in conductor.inventory.strands.collection:
            mid = strand.gauss_fields.current_along.size // 2
            value = float(np.real(strand.gauss_fields.current_along[mid]))
            if strand.identifier.startswith("STACK"):
                if self.stack_current is None:
                    self.stack_current = value
                    self.critical_current = float(
                        strand.cross_section["sc"]
                        * strand.gauss_fields.J_critical[mid]
                    )
                    # Homogenized multi-layer (Ag/Hastelloy/Cu) stabilizer
                    # resistivity - NOT pure copper.
                    self.stack_stabilizer_resistivity = float(
                        strand.gauss_fields.electrical_resistivity_stabilizer[
                            mid
                        ]
                    )
                stack = strand
            else:
                if self.external_current is None:
                    self.external_current = value
                    self.external_resistivity = float(
                        strand.gauss_fields.electrical_resistivity_stabilizer[
                            mid
                        ]
                    )
        if self.jacobian_check is None and stack is not None:
            self.jacobian_check = _finite_difference_check(conductor, stack)
        return False


def _voltage(conductor, strand, current):
    """V(I) = R(I) * I with the strand-alone resistance at bound |I|."""
    from electromagnetics.electric_solver import _bind_current_for_resistance

    _bind_current_for_resistance(strand, current)
    resistance = np.asarray(strand.get_electric_resistance(conductor))
    return resistance * current


def _finite_difference_check(conductor, strand):
    """Central-difference d(V)/d(I) vs the assembled differential
    resistance, in the SC regime (0.5 Ic) and the sharing regime (1.5 Ic).

    Returns {regime: max relative deviation}.
    """
    from electromagnetics.electric_solver import _bind_current_for_resistance

    critical = np.asarray(
        strand.cross_section["sc"] * strand.gauss_fields.J_critical
    )
    saved = np.asarray(strand.gauss_fields.current_for_resistance).copy()
    deviations = {}
    for label, fraction in (("sc", 0.5), ("sharing", 1.5)):
        current = fraction * critical
        epsilon = 1.0e-6 * current
        _bind_current_for_resistance(strand, current)
        strand.get_electric_resistance(conductor)
        derivative = np.asarray(
            strand.get_electric_resistance_derivative(conductor)
        )
        regime_index = np.asarray(
            strand._electric_regime_gauss[label], dtype=int
        )
        assert regime_index.size > 0, (
            f"expected elements in regime {label!r} at {fraction} Ic"
        )
        finite_difference = (
            _voltage(conductor, strand, current + epsilon)
            - _voltage(conductor, strand, current - epsilon)
        ) / (2.0 * epsilon)
        deviations[label] = float(
            np.abs(
                (finite_difference[regime_index] - derivative[regime_index])
                / derivative[regime_index]
            ).max()
        )
    # Restore the solver's operating point.
    _bind_current_for_resistance(strand, saved)
    strand.get_electric_resistance(conductor)
    return deviations


@pytest.fixture(scope="module")
def stiff_result(tmp_path_factory):
    total_current = STIFF_CURRENT_OVER_IC * critical_current()
    directory = tmp_path_factory.mktemp("v6_newton_stiff")
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    from simulation import Simulation

    write_run_directory(directory, True, total_current)
    capture = StiffCapture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation(str(directory) + "/", step_callback=capture).run()

    # Reference with the CAPTURED material state. The stack's stabilizer
    # is a homogenized Ag/Hastelloy/Cu mixture whose resistivity is ~2.7x
    # below pure copper; the analytic model takes one resistivity for
    # both ohmic branches, so the stack branch enters with the
    # conductance-equivalent area A_stab * rho_ext / rho_stack. At 2 Ic
    # the internal-stabilizer term is a ~3 % effect (invisible in the
    # base V6 test at 1.2 Ic).
    equivalent_stack_area = STACK_STABILIZER_AREA * (
        capture.external_resistivity / capture.stack_stabilizer_resistivity
    )
    reference = analytical.current_sharing_reference(
        total_current=total_current,
        critical_current=capture.critical_current,
        power_law_exponent=POWER_LAW_EXPONENT,
        electric_field_criterion=ELECTRIC_FIELD_CRITERION,
        stack_stabilizer_area=equivalent_stack_area,
        external_stabilizer_area=EXTERNAL_COPPER_AREA,
        copper_resistivity=capture.external_resistivity,
    )
    return total_current, reference, capture, caught


def test_newton_converges_without_cap_warning(stiff_result):
    _, _, capture, caught = stiff_result
    cap_warnings = [
        w for w in caught if "current-consistency" in str(w.message)
    ]
    assert not cap_warnings, f"Newton hit the cap: {cap_warnings[0].message}"
    assert 0 < capture.newton_iterations < 15


def test_stiff_split_matches_parallel_circuit(stiff_result):
    total_current, reference, capture, _ = stiff_result
    assert capture.stack_current is not None
    assert capture.stack_current == pytest.approx(
        reference["stack_current"], rel=0.02
    )
    assert capture.external_current == pytest.approx(
        reference["external_current"], rel=0.05
    )
    assert capture.stack_current + capture.external_current == pytest.approx(
        total_current, rel=1e-6
    )


def test_differential_resistance_matches_finite_difference(stiff_result):
    _, _, capture, _ = stiff_result
    assert capture.jacobian_check is not None
    for regime, deviation in capture.jacobian_check.items():
        assert deviation < 1.0e-4, (
            f"d(V)/d(I) mismatch in regime {regime!r}: {deviation:.3e}"
        )
