"""Run state of a conductor: clock and step control, energy/mass balances
and the diagnostics save schedule.

These attributes used to be assigned on the Conductor from the numerics,
the application loop, the energy balance and the output adapters. They are
grouped in three small state objects owned by the Conductor
(``conductor.time_state``, ``conductor.balances``, ``conductor.save_schedule``).
Fields start as UNSET so that ``getattr(conductor, name, default)`` and
``hasattr`` keep their former meaning until the attribute is first assigned;
the Conductor forwards the old spelling for one stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _Unset:
    __slots__ = ()

    def __repr__(self):
        return "UNSET"


UNSET = _Unset()


@dataclass
class TimeState:
    cond_time: Any = UNSET
    cond_num_step: Any = UNSET
    time_step: Any = UNSET
    previous_time_step: Any = UNSET
    local_truncation_error_ratio: Any = UNSET
    time_accuracy_eigenvalue: Any = UNSET
    equation_eigenvalues: Any = UNSET
    restart_after_finalization: Any = UNSET


@dataclass
class BalanceState:
    E_sol_ini: Any = UNSET
    E_sol_fin: Any = UNSET
    E_str_ini: Any = UNSET
    E_str_fin: Any = UNSET
    E_jk_ini: Any = UNSET
    E_jk_fin: Any = UNSET
    enthalpy_balance: Any = UNSET
    enthalpy_inl: Any = UNSET
    enthalpy_out: Any = UNSET
    inner_pow: Any = UNSET
    outer_pow: Any = UNSET
    mass_balance: Any = UNSET
    energy_balance: Any = UNSET
    MDTINL: Any = UNSET


@dataclass
class SaveSchedule:
    Space_save: Any = UNSET
    Time_save: Any = UNSET
    i_save: Any = UNSET
    num_step_save: Any = UNSET


# old attribute name -> name of the holder on the Conductor
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
