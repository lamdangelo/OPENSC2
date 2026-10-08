"""What a component needs to know about its conductor and the run when it is
built (replaces passing the Simulation and Conductor objects)."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.time_stepping import TimeSteppingSettings


@dataclass(frozen=True)
class ComponentBuildSettings:
    conductor_identifier: str
    conductor_length: float  # ZLENGTH [m]
    time_stepping: TimeSteppingSettings
