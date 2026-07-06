"""
This module has been superseded by the electromagnetics package.

All electric solver functions now live in:
    electromagnetics.electric_solver
    electromagnetics.operating_conditions

The re-exports below preserve backwards compatibility for any callers that
still import from this module directly.
"""

from electromagnetics.electric_solver import (
    solve_steady_state as electric_steady_state_solution,
    solve_transient as electric_transient_solution,
    assemble_solution as solution_completion,
    ELECTRIC_TIME_STEP_NUMBER,
)
from electromagnetics.operating_conditions import user_defined_current as custom_current_function

__all__ = [
    "electric_steady_state_solution",
    "electric_transient_solution",
    "solution_completion",
    "custom_current_function",
    "ELECTRIC_TIME_STEP_NUMBER",
]
