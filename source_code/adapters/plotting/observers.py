"""Plotting observers: real-time plots during the run, figures afterwards."""

from __future__ import annotations

from adapters.plotting.plots import (
    create_legend_rtp,
    create_real_time_plots,
    make_plots,
    plot_properties,
    update_real_time_plots,
)
from ports.observer import NullObserver


class RealTimePlotObserver(NullObserver):
    def on_conductor_created(self, simulation, conductor):
        create_real_time_plots(simulation, conductor)

    def on_conductor_initialized(self, simulation, conductor):
        update_real_time_plots(conductor)
        create_legend_rtp(conductor)

    def after_conductor_step(self, simulation, conductor):
        update_real_time_plots(conductor)


class FigureObserver(NullObserver):
    def on_initial_state(self, simulation, conductor):
        plot_properties(simulation, conductor)

    def on_conductor_post_processed(self, simulation, conductor):
        plot_properties(simulation, conductor, what="solution")

    def on_post_processing_done(self, simulation):
        make_plots(simulation, kind="Space_distr")
        make_plots(simulation, kind="Time_evol")
