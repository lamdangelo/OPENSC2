"""Post-processing use case (formerly Simulation.conductor_post_processing)."""

from __future__ import annotations

from opensc2.runtime.context import SimulationContext


def post_process(ctx: SimulationContext, observers, observer_target=None) -> None:
    target = ctx if observer_target is None else observer_target
    for conductor in ctx.conductors:
        # Final energies and the mass/energy balances (domain).
        conductor.post_processing(ctx)
        # Final solution, convergence data, spatial distribution files and
        # figures (observers).
        observers.on_conductor_post_processed(target, conductor)
    observers.on_post_processing_done(target)
