"""The transient use case: the time loop (formerly Simulation.conductor_solution).

Call order per step, preserved from the facade (observers are notified at
exactly the same points): callback -> time step selection -> lockstep ->
clock -> on_step_started -> per conductor: electric / operating conditions
-> heat source -> uncoupled step | coupled step -> density refresh and
enthalpy balance -> radiative exchange -> after_conductor_step -> autosave
-> on_transient_end.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from electromagnetics.electromagnetic_flags import CurrentMode
from hydraulics.hydraulic_flags import MASS_FLOW_FORMULATIONS
from numerics.coupled_step import solve_coupled_conductors_step
from numerics.transient_solution_functions import get_time_step, step
from runtime.context import SimulationContext


def run_transient(
    ctx: SimulationContext,
    observers,
    observer_target=None,
    step_callback: Optional[Callable] = None,
    callback_target=None,
) -> None:
    target = ctx if observer_target is None else observer_target
    callback_target = ctx if callback_target is None else callback_target
    settings = ctx.transient_settings
    conductors = ctx.conductors

    for conductor in conductors:
        # Radiative heat exchanged between jackets and with the environment.
        conductor.compute_radiative_heat_exhange_jk()
        conductor.compute_heat_exchange_jk_env(ctx.environment)
        observers.on_transient_start(target, conductor)
    # Any transverse insulation coupling declared on any solid forces all
    # conductors into lockstep time stepping.
    transverse_coupling_declared = any(
        getattr(solid.operations, "transverse_coupling_file", "")
        for conductor in conductors
        for solid in conductor.inventory.solids.collection
    )
    while ctx.simulation_time[-1] < settings["TEND"] - 1e-5 * settings["STPMIN"]:
        if step_callback is not None and step_callback(callback_target):
            # The driver requested the end of the transient.
            break
        ctx.num_step = ctx.num_step + 1
        time_step = np.zeros(len(conductors))
        # Conductors coupled to the hydraulic network advance together with
        # it (one shared linear solve per step).
        coupled_conductors = [c for c in conductors if c.network_ports]
        for conductor in conductors:
            get_time_step(conductor, settings, ctx.num_step)
        # Network-coupled conductors share the network unknowns and must
        # advance with a common time step; with any transverse coupling
        # declared, every conductor shares the smallest step.
        lockstep_conductors = list(conductors) if transverse_coupling_declared else coupled_conductors
        if lockstep_conductors:
            shared_time_step = min(c.time_step for c in lockstep_conductors)
            for conductor in lockstep_conductors:
                conductor.time_state.time_step = shared_time_step
        for ii, conductor in enumerate(conductors):
            time_step[ii] = conductor.time_state.time_step
            conductor.time_state.cond_time.append(conductor.time_state.cond_time[-1] + conductor.time_state.time_step)
            conductor.time_state.cond_num_step = conductor.time_state.cond_num_step + 1
        ctx.simulation_time_step = np.max(time_step)
        ctx.simulation_time.append(ctx.simulation_time[-1] + ctx.simulation_time_step)
        observers.on_step_started(target)
        for conductor in conductors:
            if conductor.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
                conductor.electric_method()
            else:
                # Solid component properties are updated in
                # operating_conditions_em (called by electric_method otherwise).
                conductor.operating_conditions_em()
            # Thermal-hydraulic properties in the Gauss points before the
            # implicit step.
            conductor.operating_conditions_th(ctx)
            conductor.build_heat_source(ctx)
            # Conductors coupled to the hydraulic network are solved jointly
            # with the network below, once every coupled conductor is prepared.
            if not conductor.network_ports:
                step(
                    conductor,
                    ctx.environment,
                    ctx.qsources[conductor.identifier],
                    ctx.num_step,
                    ctx.diagnostics,
                )
        if coupled_conductors:
            # One monolithic solve of all network-coupled conductors and the
            # network; advances the network state exactly once.
            solve_coupled_conductors_step(
                coupled_conductors, ctx.hydraulic_network, ctx.qsources, ctx.diagnostics
            )
        for conductor in conductors:
            for fluid_comp in conductor.inventory.fluids.collection:
                # Nodal density and mass flow rate from the updated state.
                if conductor.hydraulic_formulation in MASS_FLOW_FORMULATIONS:
                    fluid_comp.coolant._refresh_density_and_velocity_from_mass_flow()
                else:
                    fluid_comp.coolant._compute_density_and_mass_flow_rates_nodal_gauss(
                        conductor
                    )
                # Enthalpy balance: sum((mdot*w)_out - (mdot*w)_inl).
                fields = fluid_comp.coolant.node_fields
                conductor.balances.enthalpy_balance = conductor.balances.enthalpy_balance + conductor.time_state.time_step * (
                    fields.mass_flow_rate[-1] * fields.total_enthalpy[-1]
                    - fields.mass_flow_rate[0] * fields.total_enthalpy[0]
                )
                conductor.balances.enthalpy_out = (
                    conductor.balances.enthalpy_out
                    + conductor.time_state.time_step * fields.mass_flow_rate[-1] * fields.total_enthalpy[-1]
                )
                conductor.balances.enthalpy_inl = (
                    conductor.balances.enthalpy_inl
                    + conductor.time_state.time_step * fields.mass_flow_rate[0] * fields.total_enthalpy[0]
                )
            conductor.compute_radiative_heat_exhange_jk()
            conductor.compute_heat_exchange_jk_env(ctx.environment)
            observers.after_conductor_step(target, conductor)

        autosave_interval = settings["AUTOSAVE_INTERVAL"]
        if autosave_interval is not None and ctx.num_step % autosave_interval == 0:
            observers.on_autosave(target)
    observers.on_transient_end(target)
