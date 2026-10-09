"""Run state of a conductor: clock and step control, energy/mass balances
and the diagnostics save schedule.

These attributes used to be assigned on the Conductor from the numerics,
the application loop, the energy balance and the output adapters. They are
grouped in three small state objects owned by the Conductor
(``conductor.time_state``, ``conductor.balances``, ``conductor.save_schedule``)
and are always read and written through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TimeState:
    cond_time: list = field(default_factory=list)  # simulation time history [s]
    cond_num_step: int = 0
    time_step: Optional[float] = None
    previous_time_step: Optional[float] = None
    local_truncation_error_ratio: Any = None
    time_accuracy_eigenvalue: Any = None
    equation_eigenvalues: Any = None
    restart_after_finalization: bool = False


@dataclass
class BalanceState:
    E_sol_ini: Any = None
    E_sol_fin: Any = None
    E_str_ini: Any = None
    E_str_fin: Any = None
    E_jk_ini: Any = None
    E_jk_fin: Any = None
    enthalpy_balance: Any = None
    enthalpy_inl: Any = None
    enthalpy_out: Any = None
    inner_pow: Any = None
    outer_pow: Any = None
    mass_balance: Any = None
    energy_balance: Any = None
    MDTINL: Any = None


@dataclass
class SaveSchedule:
    Space_save: Any = None  # times at which spatial distributions are saved
    Time_save: Any = None  # positions at which time evolutions are saved
    i_save: Optional[int] = None  # number of spatial saves performed so far
    num_step_save: Any = None  # step numbers of the scheduled spatial saves


# former Conductor attribute name -> name of its holder on the Conductor
# (kept as documentation and for tests/test_architecture.py, which fails on
# any read of the old spelling)
RUN_STATE_ATTRIBUTES = {
    "cond_time": "time_state",
    "cond_num_step": "time_state",
    "time_step": "time_state",
    "previous_time_step": "time_state",
    "local_truncation_error_ratio": "time_state",
    "time_accuracy_eigenvalue": "time_state",
    "equation_eigenvalues": "time_state",
    "restart_after_finalization": "time_state",
    "E_sol_ini": "balances",
    "E_sol_fin": "balances",
    "E_str_ini": "balances",
    "E_str_fin": "balances",
    "E_jk_ini": "balances",
    "E_jk_fin": "balances",
    "enthalpy_balance": "balances",
    "enthalpy_inl": "balances",
    "enthalpy_out": "balances",
    "inner_pow": "balances",
    "outer_pow": "balances",
    "mass_balance": "balances",
    "energy_balance": "balances",
    "MDTINL": "balances",
    "Space_save": "save_schedule",
    "Time_save": "save_schedule",
    "i_save": "save_schedule",
    "num_step_save": "save_schedule",
}
