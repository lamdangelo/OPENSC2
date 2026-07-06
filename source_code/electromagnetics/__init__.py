"""
The electromagnetics package provides all modules for the electric problem in
the OPENSC2 conductor simulation.

Module overview
---------------
electromagnetic_flags
    Enumerations for all electromagnetics operating modes (current mode,
    inductance mode, electric solver type, conductance mode, B-field definition).

circuit_topology
    Element-to-node connectivity and incidence matrices for the current-carrying
    strand components, and transverse contact detection.

resistance
    Diagonal electric resistance matrix from per-element component resistivities.

inductance
    Full inductance matrix (self + mutual) via analytical Neumann formula or
    numerical approximation.

conductance
    Transverse electric conductance matrix between contacting strand components.

system_assembly
    Assembly of the global stiffness matrix, mass matrix, known-term vector,
    and right-hand side for the electric linear system.

boundary_conditions
    Application of Dirichlet boundary conditions: equipotential surfaces and
    fixed-potential nodes.

electric_solver
    Steady-state and transient sparse solvers; post-processing (current and
    potential distribution, Joule power, voltage sums).

operating_conditions
    Per-time-step updates of transport current, magnetic field, critical
    properties, and electrical resistivity.
"""

from importlib import import_module

# Convenience re-exports are provided lazily (PEP 562). Importing them eagerly
# here pulled in operating_conditions / electric_solver at package-init time,
# which import solid-component classes and created a circular import whenever a
# component module imported ``electromagnetics.electromagnetic_flags`` (importing
# any submodule runs this __init__ first). Deferring the imports until an
# attribute is actually accessed off the package breaks that cycle while keeping
# ``from electromagnetics import <name>`` working for consumers.
_LAZY_EXPORTS: dict[str, str] = {
    # Flags
    "BFieldDefinitionType": "electromagnetic_flags",
    "CurrentMode": "electromagnetic_flags",
    "InductanceMode": "electromagnetic_flags",
    "SelfInductanceMode": "electromagnetic_flags",
    "ElectricSolver": "electromagnetic_flags",
    "ElectricConductanceMode": "electromagnetic_flags",
    # Circuit topology
    "build_connectivity_current_carriers": "circuit_topology",
    "build_incidence_matrix": "circuit_topology",
    "detect_contacts": "circuit_topology",
    "build_contact_incidence_matrix": "circuit_topology",
    # Matrix builders
    "build_resistance_matrix": "resistance",
    "build_inductance_matrix": "inductance",
    "evaluate_transversal_distance": "conductance",
    "evaluate_electric_conductance": "conductance",
    "build_conductance_matrix": "conductance",
    # System assembly
    "build_stiffness_matrix": "system_assembly",
    "build_mass_matrix": "system_assembly",
    "build_known_term_vector": "system_assembly",
    "build_right_hand_side": "system_assembly",
    # Boundary conditions
    "assign_equipotential_surfaces": "boundary_conditions",
    "assign_fixed_potential": "boundary_conditions",
    "reduce_system": "boundary_conditions",
    # Solvers
    "solve_steady_state": "electric_solver",
    "solve_transient": "electric_solver",
    "assemble_solution": "electric_solver",
    "reorganize_solution": "electric_solver",
    "evaluate_joule_power_conductance": "electric_solver",
    "compute_voltage_sum": "electric_solver",
    "ELECTRIC_TIME_STEP_NUMBER": "electric_solver",
    # Operating conditions
    "update_em_operating_conditions": "operating_conditions",
    "evaluate_em_gauss_points": "operating_conditions",
    "evaluate_total_operating_current": "operating_conditions",
    "user_defined_current": "operating_conditions",
}


def __getattr__(name: str):
    """Lazily import a re-exported name from its submodule (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_EXPORTS))


__all__ = [
    # Flags
    "BFieldDefinitionType",
    "CurrentMode",
    "InductanceMode",
    "SelfInductanceMode",
    "ElectricSolver",
    "ElectricConductanceMode",
    # Circuit topology
    "build_connectivity_current_carriers",
    "build_incidence_matrix",
    "detect_contacts",
    "build_contact_incidence_matrix",
    # Matrix builders
    "build_resistance_matrix",
    "build_inductance_matrix",
    "evaluate_transversal_distance",
    "evaluate_electric_conductance",
    "build_conductance_matrix",
    # System assembly
    "build_stiffness_matrix",
    "build_mass_matrix",
    "build_known_term_vector",
    "build_right_hand_side",
    # Boundary conditions
    "assign_equipotential_surfaces",
    "assign_fixed_potential",
    "reduce_system",
    # Solvers
    "solve_steady_state",
    "solve_transient",
    "assemble_solution",
    "reorganize_solution",
    "evaluate_joule_power_conductance",
    "compute_voltage_sum",
    "ELECTRIC_TIME_STEP_NUMBER",
    # Operating conditions
    "update_em_operating_conditions",
    "evaluate_em_gauss_points",
    "evaluate_total_operating_current",
    "user_defined_current",
]
