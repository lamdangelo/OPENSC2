"""Initialization use cases: instantiate the conductors and bring them to
their initial state (formerly Simulation.conductor_instance and
Simulation.conductor_initialization). Depends on the domain and the ports
only; the facade injects the definition loader and the observers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from conductor.conductor import Conductor
from conductor.conductor_definition import ConductorDefinition
from electromagnetics.electromagnetic_flags import CurrentMode
from hydraulics.formulation import resolve_hydraulic_formulation
from hydraulics.hydraulic_flags import HydraulicFormulation
from hydraulics.network.coupling import build_coupled_network
from runtime.context import SimulationContext


def instantiate_conductors(
    ctx: SimulationContext,
    conductor_count: int,
    load_definition: Callable[[int], ConductorDefinition],
    conductor_contacts,
    observers,
    observer_target=None,
) -> None:
    """Build every conductor from its definition and wire its geometry."""
    target = ctx if observer_target is None else observer_target
    for counter in range(1, 1 + conductor_count):
        conductor = Conductor(load_definition(counter))
        conductor.initialize_with_simulation(ctx)
        ctx.conductors.append(conductor)
    ctx.conductor_contacts = conductor_contacts
    for conductor in ctx.conductors:
        observers.on_conductor_created(target, conductor)


def initialize_conductors(
    ctx: SimulationContext,
    network_mapping: Optional[dict],
    observers,
    observer_target=None,
    report=print,
) -> None:
    """Initial state of every conductor, the heat-source containers and the
    optional hydraulic network."""
    target = ctx if observer_target is None else observer_target
    # Resolve the declared hydraulic formulation of every conductor before
    # any initialization work: the solution seeding inside
    # cond.initialization already depends on it. A conductor counts as
    # network-coupled when the (YAML-only) hydraulic_network section
    # declares a port on it; the network itself is built further below.
    ported_conductor_identifiers = set()
    if network_mapping is not None:
        ported_conductor_identifiers = {
            port.get("conductor") for port in network_mapping.get("ports", [])
        }
    for cond in ctx.conductors:
        coupling_enabled = cond.identifier in ported_conductor_identifiers
        cond.set_hydraulic_formulation(
            resolve_hydraulic_formulation(
                cond.inputs.hydraulic_formulation,
                coupling_enabled,
                cond.inputs.explicit_mass_flow_formulation,
            )
        )
        declared = cond.inputs.hydraulic_formulation
        reason = (
            f"auto: network coupling {'declared' if coupling_enabled else 'absent'}"
            if declared is HydraulicFormulation.AUTO
            else "explicit input"
        )
        report(
            f"Conductor {cond.identifier}: hydraulic formulation "
            f"'{cond.hydraulic_formulation.value}' ({reason}).\n"
        )
    for cond in ctx.conductors:
        # ** INITIALIZATION **
        ctx.simulation_time = [0.0]
        ctx.num_step = 0
        cond.initialization(ctx)
        observers.on_conductor_initialized(target, cond)
        # Use the electric method only if needed, i.e. the user specifies a
        # current.
        if cond.inputs.current_mode != CurrentMode.CURRENT_NOT_DEFINED:
            cond.eval_total_operating_current()
            cond.electric_method()
        else:
            # Solid component thermophysical, electromagnetic and critical
            # properties are updated in operating_conditions_em, which
            # electric_method calls in all the other cases.
            cond.operating_conditions_em()
        observers.on_initial_state(target, cond)
        # ** END INITIALIZATION **

    # Heat-source containers (one per conductor; the number of columns is
    # the number of SolidComponent equations, see numerics build_svec).
    ctx.qsources = dict()
    number = len(ctx.conductors)
    if number == 1:
        ctx.qsources[cond.identifier] = np.zeros(
            (cond.mesh.number_of_nodes, cond.equation_counts.solid_equations)
        )
    else:
        contacts = np.asarray(ctx.conductor_contacts)
        for rr in range(number):
            cond_r = ctx.conductors[rr]
            if all(contacts[rr, :]) == 0:
                # No contact between cond_r and all the others.
                ctx.qsources[cond_r.identifier] = np.zeros(
                    (cond_r.mesh.number_of_nodes, cond_r.equation_counts.solid_equations)
                )
            else:
                for cc in range(rr + 1, number):
                    cond_c = ctx.conductors[cc]
                    if contacts[rr, cc] == 1:
                        raise ValueError(
                            "Multi conductors with thermal contact: not yet managed the situation, need to build dict_qsource properly!!"
                        )
                    else:
                        ctx.qsources[f"{cond_r.identifier}_{cond_c.identifier}"] = np.zeros(
                            (cond_r.mesh.number_of_nodes, cond_r.equation_counts.solid_equations)
                        )
                        raise ValueError(
                            "Multi conductors with thermal contact: not yet managed the situation, need to build dict_qsource properly!!"
                        )

    # Build, resolve and initialize the optional hydraulic network coupled
    # to conductor channel ends. Must run after the conductor loop above:
    # the port resolution reuses the boundary-condition equation indices and
    # the initial flow built there.
    if network_mapping is not None:
        ctx.hydraulic_network = build_coupled_network(ctx, network_mapping)
        observers.on_network_built(target)
