"""
This module manages the electromagnetic operating conditions that must be
re-evaluated at each electric time step:

* Applied transport current (from constant value, external file, or custom
  function).
* Magnetic field and its axial gradient at nodal and Gauss-point locations.
* Superconductor critical properties (Jc, Tc, TCS) for mixed and stack
  strand components.
* Electrical resistivity of the stabiliser material.

:func:`update_em_operating_conditions` is the top-level function called
once per electric time step.  It dispatches to
:func:`evaluate_em_gauss_points` for Gauss-point quantities.

:func:`evaluate_total_operating_current` builds the per-conductor current
boundary-condition vector from the active ``CurrentMode``.

:func:`user_defined_current` is the user-facing hook for defining an
arbitrary time-dependent transport current waveform.

Ported from the Conductor class methods:
    operating_conditions_em
    __eval_gauss_point_em
    eval_total_operating_current
From utility_functions/electric_auxiliary_functions.py:
    custom_current_function
"""

from typing import Union

import numpy as np

from electromagnetics.electromagnetic_flags import CurrentMode
from components.solid.strand_stabilizer_component import StrandStabilizerComponent
from components.solid.strand_mixed_component import StrandMixedComponent
from thermal.temperature_field import eval_temperature_solids_gauss_point


# ---------------------------------------------------------------------------
# USER-CUSTOMISABLE CURRENT WAVEFORM
# ---------------------------------------------------------------------------

def user_defined_current(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the transport current at the given time(s).

    **Customise this function** to define an arbitrary current waveform.
    It is called when ``CurrentMode.CURRENT_IS_FUNCTION`` is active.

    Args:
        time: Scalar or array of time values in seconds.

    Returns:
        Current value(s) in amperes at the requested time(s).
    """
    CURRENT_AMPLITUDE = 1.0  # [A]
    FREQUENCY = 50.0         # [Hz]
    return CURRENT_AMPLITUDE * np.cos(2 * np.pi * FREQUENCY * time)


# ---------------------------------------------------------------------------
# OPERATING-CONDITION UPDATES
# ---------------------------------------------------------------------------

def update_em_operating_conditions(conductor: object) -> None:
    """Update all electromagnetic operating conditions for the current sub-step.

    For every strand component:
    * Retrieves the transport current from the boundary condition.
    * Loads the magnetic field and its axial gradient.
    * For superconducting strands (not pure stabiliser): evaluates strain
      (Nb3Sn only, at step ≤ 1), computes critical properties (Jc, Tc),
      and — depending on ``tcs_evaluation`` — evaluates the current-sharing
      temperature.
    * Updates solid-component electrical resistivity.

    For every jacket component:
    * Retrieves the transport current and magnetic field.
    * Updates electrical resistivity.

    Finishes by delegating to :func:`evaluate_em_gauss_points`.

    Args:
        conductor: Conductor object at the current electric time step.
    """
    for strand in conductor.inventory.strands.collection:
        strand.get_current(conductor)
        strand.get_magnetic_field(conductor)
        strand.get_magnetic_field_gradient(conductor)

        if not isinstance(strand, StrandStabilizerComponent):
            if strand.inputs.superconducting_material == "nb3sn":
                if conductor.cond_el_num_step <= 1:
                    strand.get_eps(conductor)
            strand.get_superconductor_critical_prop(conductor)
            if (
                strand.operations.tcs_evaluation == False
                and conductor.cond_num_step == 0
            ):
                strand.get_tcs()
            elif strand.operations.tcs_evaluation == True:
                if conductor.cond_el_num_step <= 1:
                    strand.get_tcs()

        if conductor.cond_el_num_step <= 1:
            strand.eval_sol_comp_properties(conductor.inventory)
        else:
            _update_electrical_resistivity_nodal(strand)

    for jacket in conductor.inventory.jackets.collection:
        jacket.get_current(conductor)
        jacket.get_magnetic_field(conductor)
        if conductor.cond_el_num_step <= 1:
            jacket.eval_sol_comp_properties(conductor.inventory)
        else:
            jacket.node_fields.total_electrical_resistivity = (
                jacket.jacket_electrical_resistivity(jacket.node_fields)
            )

    evaluate_em_gauss_points(conductor)


def evaluate_em_gauss_points(conductor: object) -> None:
    """Evaluate electromagnetic quantities at element Gauss points.

    Temperature at Gauss points is interpolated from the nodal values first.
    Then, for jackets and strands, the magnetic field, strain, critical
    properties, and electrical resistivity are re-evaluated at the Gauss
    points.

    Args:
        conductor: Conductor object at the current electric time step.
    """
    eval_temperature_solids_gauss_point(conductor)

    for jacket in conductor.inventory.jackets.collection:
        jacket.get_magnetic_field(conductor, nodal=False)
        if conductor.cond_el_num_step <= 1:
            jacket.eval_sol_comp_properties(conductor.inventory, nodal=False)
        else:
            jacket.gauss_fields.total_electrical_resistivity = (
                jacket.jacket_electrical_resistivity(jacket.gauss_fields)
            )

    for strand in conductor.inventory.strands.collection:
        strand.get_magnetic_field(conductor, nodal=False)
        strand.get_magnetic_field_gradient(conductor, nodal=False)

        if not isinstance(strand, StrandStabilizerComponent):
            if strand.inputs.superconducting_material == "nb3sn":
                if conductor.cond_el_num_step <= 1:
                    strand.get_eps(conductor, nodal=False)
            strand.get_superconductor_critical_prop(conductor, nodal=False)
            if (
                strand.operations.tcs_evaluation == False
                and conductor.cond_num_step == 0
            ):
                strand.get_tcs(nodal=False)
            elif strand.operations.tcs_evaluation == True:
                if conductor.cond_el_num_step <= 1:
                    strand.get_tcs(nodal=False)

        if conductor.cond_el_num_step <= 1:
            strand.eval_sol_comp_properties(conductor.inventory, nodal=False)
        else:
            _update_electrical_resistivity_gauss(strand)


def evaluate_total_operating_current(conductor: object) -> None:
    """Build the nodal operating-current vector for the current sub-step.

    The boundary condition for the electric problem is a current source at
    the first node of the first current carrier (inlet) and an equal current
    sink at the last node of the last current carrier (outlet).

    The mode is read from ``conductor.inputs.current_mode``:

    * ``CURRENT_IS_CONSTANT``: constant value ``conductor.inputs.initial_current``.
    * ``CURRENT_IS_FROM_FILE``: summed from per-strand nodal current values
      already loaded from an external file into each solid component's
      ``node_fields.op_current``.
    * ``CURRENT_IS_FUNCTION``: evaluated from :func:`user_defined_current` at
      the current electric time step.

    Result stored in ``conductor.node_fields.op_current``.

    Args:
        conductor: Conductor object with inputs, inventory, and electric
            time step set up.
    """
    conductor.node_fields.op_current = np.zeros(
        conductor.total_nodes_current_carriers
    )

    mode = conductor.inputs.current_mode

    if mode == CurrentMode.CURRENT_IS_CONSTANT:
        conductor.node_fields.op_current[0] = conductor.inputs.initial_current
        conductor.node_fields.op_current[-1] = -conductor.inputs.initial_current

    elif mode == CurrentMode.CURRENT_IS_FROM_FILE:
        for solid in conductor.inventory.solids.collection:
            conductor.node_fields.op_current[0] += (
                solid.node_fields.op_current[0]
            )
            conductor.node_fields.op_current[-1] += (
                solid.node_fields.op_current[-1]
            )
        # Outlet current is exiting — flip sign
        conductor.node_fields.op_current[-1] = (
            -conductor.node_fields.op_current[-1]
        )

    elif mode == CurrentMode.CURRENT_IS_FUNCTION:
        I = user_defined_current(conductor.electric_time_step)
        conductor.node_fields.op_current[0] = I
        conductor.node_fields.op_current[-1] = -I


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _update_electrical_resistivity_nodal(strand: object) -> None:
    """Update only the stabiliser electrical resistivity at nodal points.

    Called at electric time steps > 1 when a full property evaluation is not
    needed (only resistivity changes with temperature at each sub-step).

    Args:
        strand: A StrandMixedComponent or StrandStabilizerComponent object.
    """
    if isinstance(strand, StrandMixedComponent):
        strand.node_fields.electrical_resistivity_stabilizer = (
            strand.strand_electrical_resistivity_not_sc(strand.node_fields)
        )
    elif isinstance(strand, StrandStabilizerComponent):
        strand.node_fields.electrical_resistivity_stabilizer = (
            strand.strand_electrical_resistivity(strand.node_fields)
        )


def _update_electrical_resistivity_gauss(strand: object) -> None:
    """Update only the stabiliser electrical resistivity at Gauss points.

    Args:
        strand: A StrandMixedComponent or StrandStabilizerComponent object.
    """
    if isinstance(strand, StrandMixedComponent):
        strand.gauss_fields.electrical_resistivity_stabilizer = (
            strand.strand_electrical_resistivity_not_sc(strand.gauss_fields)
        )
    elif isinstance(strand, StrandStabilizerComponent):
        strand.gauss_fields.electrical_resistivity_stabilizer = (
            strand.strand_electrical_resistivity(strand.gauss_fields)
        )
