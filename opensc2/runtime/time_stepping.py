"""Time-stepping settings needed by domain objects at construction time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TimeSteppingSettings:
    adaptive_mode: int  # IADAPTIME (0 = fixed step)
    minimum_time_step: float  # STPMIN [s]

    @classmethod
    def from_transient_settings(cls, transient_settings: Mapping[str, Any]):
        return cls(
            adaptive_mode=transient_settings["IADAPTIME"],
            minimum_time_step=transient_settings["STPMIN"],
        )
