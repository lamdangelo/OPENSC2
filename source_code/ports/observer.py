"""Simulation observer port.

Everything observable about a run (result files, plots, the summary log,
checkpoints, console progress) happens in observers notified by the
application at fixed points; the model and the numerics never write or
draw. The hook names follow the phases of a run; the Simulation facade
calls them in the order documented in its loop.
"""

from __future__ import annotations

from typing import Protocol


class SimulationObserver(Protocol):
    # -- setup -------------------------------------------------------------
    def on_conductor_created(self, simulation, conductor) -> None: ...

    def on_conductor_initialized(self, simulation, conductor) -> None:
        """Conductor.initialization finished (geometry and initial fields)."""
        ...

    def on_initial_state(self, simulation, conductor) -> None:
        """Initial electric/thermal state of the conductor established."""
        ...

    def on_network_built(self, simulation) -> None: ...

    def on_transient_start(self, simulation, conductor) -> None: ...

    # -- time loop ---------------------------------------------------------
    def on_step_started(self, simulation) -> None: ...

    def after_conductor_step(self, simulation, conductor) -> None: ...

    def on_autosave(self, simulation) -> None: ...

    def on_transient_end(self, simulation) -> None: ...

    # -- post-processing ---------------------------------------------------
    def on_conductor_post_processed(self, simulation, conductor) -> None: ...

    def on_post_processing_done(self, simulation) -> None: ...

    def on_finished(self, simulation) -> None: ...


class NullObserver:
    """Base class: every hook is a no-op; observers override what they need."""

    def on_conductor_created(self, simulation, conductor): pass
    def on_conductor_initialized(self, simulation, conductor): pass
    def on_initial_state(self, simulation, conductor): pass
    def on_network_built(self, simulation): pass
    def on_transient_start(self, simulation, conductor): pass
    def on_step_started(self, simulation): pass
    def after_conductor_step(self, simulation, conductor): pass
    def on_autosave(self, simulation): pass
    def on_transient_end(self, simulation): pass
    def on_conductor_post_processed(self, simulation, conductor): pass
    def on_post_processing_done(self, simulation): pass
    def on_finished(self, simulation): pass
