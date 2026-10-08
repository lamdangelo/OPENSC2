"""Electric network state of a conductor.

The ~35 arrays, matrices and counters that the electromagnetics package
builds and advances used to be loose attributes assigned on the Conductor
from outside the class. They are grouped here; ``Conductor.electric`` holds
the instance and physics reads/writes ``conductor.electric.<name>``. The
old ``conductor.<name>`` spelling is forwarded by the Conductor for one
stage (see ELECTRIC_ATTRIBUTES) and removed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ELECTRIC_ATTRIBUTES = (
    "electric_known_term_vector",
    "electric_known_term_vector_old",
    "electric_solution",
    "electric_solution_steady",
    "electric_stiffness_matrix",
    "electric_right_hand_side",
    "electric_mass_matrix",
    "electric_resistance_matrix",
    "electric_differential_resistance_matrix",
    "electric_conductance_matrix",
    "electric_conductance_diag_matrix",
    "electric_conductance",
    "electric_conductance_mode",
    "electric_reduction_operator",
    "electric_retained_index",
    "electric_theta",
    "electric_time",
    "electric_time_end",
    "electric_time_step",
    "inductance_matrix",
    "incidence_matrix",
    "incidence_matrix_transposed",
    "contact_incidence_matrix",
    "contact_nodes_current_carriers",
    "_contact_nodes_first",
    "fixed_potential_index",
    "fixed_potential_value",
    "equipotential_node_index",
    "nodal_potential",
    "current_across",
    "delta_voltage_across",
    "_steady_state_permutation",
    "_steady_newton_iterations",
    "cond_el_num_step",
    "parallel_jacket_pairs",
    "total_elements_current_carriers",
)


@dataclass
class ElectricState:
    # Topology
    incidence_matrix: Any = None
    incidence_matrix_transposed: Any = None
    contact_incidence_matrix: Any = None
    contact_nodes_current_carriers: Any = None
    _contact_nodes_first: Any = None
    total_elements_current_carriers: int = 0
    parallel_jacket_pairs: Any = field(default_factory=list)  # absent == no pairs
    # Coefficient matrices
    electric_resistance_matrix: Any = None
    electric_differential_resistance_matrix: Any = None
    electric_conductance: Any = None
    electric_conductance_mode: Any = None
    electric_conductance_matrix: Any = None
    electric_conductance_diag_matrix: Any = None
    inductance_matrix: Any = None
    electric_stiffness_matrix: Any = None
    electric_mass_matrix: Any = None
    # Boundary conditions / reduction
    fixed_potential_index: Any = None
    fixed_potential_value: Any = None
    equipotential_node_index: Any = None
    electric_reduction_operator: Any = None
    electric_retained_index: Any = None
    # Right-hand side and solution
    electric_known_term_vector: Any = None
    electric_known_term_vector_old: Any = None
    electric_right_hand_side: Any = None
    electric_solution: Any = None
    electric_solution_steady: Any = None
    nodal_potential: Any = None
    current_across: Any = None
    delta_voltage_across: Any = None
    _steady_state_permutation: Any = None
    _steady_newton_iterations: Optional[int] = None
    # Electric clock
    electric_theta: float = 0.0
    electric_time: float = 0.0
    electric_time_end: float = 0.0
    electric_time_step: float = 0.0
    cond_el_num_step: int = 0
