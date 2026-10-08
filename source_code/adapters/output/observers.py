"""Output-side observers: result files, summary log, checkpoints, console.

Each hook wraps the writer functions exactly as the Simulation facade used
to call them, in the same order, so the produced files are unchanged.
"""

from __future__ import annotations

from adapters.output.checkpoint import write_checkpoint
from adapters.output.simulation_log import SimulationSummary
from adapters.output.solution_writer import (
    reorganize_spatial_distribution,
    save_convergence_data,
    save_geometry_discretization,
    save_network_simulation_time,
    save_properties,
    save_simulation_space,
    save_simulation_time,
)
from ports.observer import NullObserver


class OutputObserver(NullObserver):
    """Result files, hot-spot/summary log and autosave checkpoints."""

    def __init__(self, summary: SimulationSummary):
        self.summary = summary

    def on_conductor_initialized(self, simulation, conductor):
        save_geometry_discretization(
            conductor.inventory.all_components.collection,
            simulation.dict_path[f"Output_Initialization_{conductor.identifier}_dir"],
        )
        save_properties(
            conductor,
            simulation.dict_path[f"Output_Initialization_{conductor.identifier}_dir"],
        )

    def on_initial_state(self, simulation, conductor):
        save_simulation_time(simulation, conductor)
        if not simulation.transient_input["RESTART"]:
            # Seed the summary extrema and hot-spot log with the t = 0
            # state; on restart the seeding happens after the checkpoint
            # restore instead.
            self.summary.update(simulation, conductor)

    def on_network_built(self, simulation):
        # Header of the network time-evolution file and initial state
        # (num_step is still 0 here).
        for conductor in simulation.list_of_Conductors:
            save_network_simulation_time(simulation, conductor)

    def on_transient_start(self, simulation, conductor):
        # On restart the t = 0 save already exists and i_save points at the
        # next pending save time.
        if not simulation.transient_input["RESTART"]:
            save_simulation_space(
                conductor,
                simulation.dict_path[
                    f"Output_Spatial_distribution_{conductor.identifier}_dir"
                ],
                abs(simulation.n_digit_time),
            )

    def after_conductor_step(self, simulation, conductor):
        if (
            conductor.i_save < len(conductor.Space_save) - 1
            and abs(conductor.cond_time[-1] - conductor.Space_save[conductor.i_save])
            < conductor.time_step / 2.0
        ):
            # Spatial distribution at the user-defined save times.
            save_simulation_space(
                conductor,
                simulation.dict_path[
                    f"Output_Spatial_distribution_{conductor.identifier}_dir"
                ],
                abs(simulation.n_digit_time),
            )
        self.summary.update(simulation, conductor)
        save_simulation_time(simulation, conductor)
        # Hydraulic network state (no-op without network ports).
        save_network_simulation_time(simulation, conductor)

    def on_autosave(self, simulation):
        # Flush the time-evolution buffers first, so the output files are
        # complete up to the checkpoint time and the checkpoint itself never
        # has to carry buffered rows.
        for conductor in simulation.list_of_Conductors:
            save_simulation_time(simulation, conductor, flush_only=True)
            save_network_simulation_time(simulation, conductor, flush_only=True)
        self.summary.flush_hotspots(simulation)
        write_checkpoint(simulation)

    def on_transient_end(self, simulation):
        # Final flush of the partially filled time-evolution buffers: a run
        # stopped before TEND never satisfies the TEND-based flush condition.
        for conductor in simulation.list_of_Conductors:
            save_simulation_time(simulation, conductor, flush_only=True)
            save_network_simulation_time(simulation, conductor, flush_only=True)

    def on_conductor_post_processed(self, simulation, conductor):
        # Space and time convergence data (solution at TEND with the mass
        # and energy balances).
        save_convergence_data(conductor, simulation.dict_path["Space_conv_output_dir"])
        save_convergence_data(
            conductor,
            simulation.dict_path["Time_conv_output_dir"],
            abs(simulation.n_digit_time),
            space_conv=False,
        )
        save_simulation_space(
            conductor,
            simulation.dict_path[f"Output_Spatial_distribution_{conductor.identifier}_dir"],
            abs(simulation.n_digit_time),
        )
        save_properties(
            conductor, simulation.dict_path[f"Output_Solution_{conductor.identifier}_dir"]
        )
        print("Saved final solution\n")
        reorganize_spatial_distribution(
            conductor,
            simulation.dict_path[f"Output_Spatial_distribution_{conductor.identifier}_dir"],
            simulation.n_digit_time,
        )

    def on_finished(self, simulation):
        self.summary.write(simulation)  # simulation.log + hot-spot flush


class ConsoleProgressObserver(NullObserver):
    def on_step_started(self, simulation):
        print(
            f"Simulation time: {simulation.simulation_time[-1]:.{simulation.n_digit_time}f} s; "
            f"{simulation.simulation_time[-1]/simulation.transient_input['TEND']*100:5.2f} %"
        )

    def on_transient_end(self, simulation):
        print("End simulation called " + simulation.transient_input["SIMULATION"] + "\n")
