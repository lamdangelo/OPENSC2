"""Architecture ratchet: layered dependencies of ``source_code``.

Target architecture (docs/architecture_refactoring_plan.md): concentric layers
whose dependencies point inward only.

    0 domain       physics, numerics, typed inputs, ports (Protocols)
    1 application  use cases (initialization, time loop, restart, observers)
    2 adapters     input readers, output writers, plotting, property libraries
    3 drivers      Simulation facade, GUI, CLI

Because ``pyproject.toml`` maps every subfolder of ``source_code`` to a top
level package, the layer of a module is decided by its top-level package
(``LAYER_OF``). Three ratchets are enforced, each against a committed
baseline of today's violations:

* ``BASELINE_LAYER_VIOLATIONS``: imports from an inner layer to an outer one,
  or of a forbidden external library (``FORBIDDEN_EXTERNAL``), as
  ``"source_module -> target"``;
* ``BASELINE_EXTERNAL_CONDUCTOR_ATTRIBUTES``: attributes assigned on a
  ``conductor``/``cond`` object outside the ``conductor`` package, as
  ``"module: attribute"``;
* ``BASELINE_DEFERRED_IMPORTS``: function-local imports of internal modules
  (the way import cycles are currently hidden), as ``"module -> target"``.

Each test fails both when a *new* violation appears and when a baseline entry
is no longer a violation (stale entries must be deleted, so the baseline can
only shrink). Run ``python tests/test_architecture.py`` to print the current
sets in a form that can be pasted into this file.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "source_code"

# --------------------------------------------------------------------------
# Layer assignment
# --------------------------------------------------------------------------

DOMAIN = 0
APPLICATION = 1
ADAPTERS = 2
DRIVERS = 3

LAYER_OF: dict[str, int] = {
    # domain
    "physical_fields": DOMAIN,
    "properties_of_materials": DOMAIN,
    "hydraulics": DOMAIN,
    "thermal": DOMAIN,
    "electromagnetics": DOMAIN,
    "components": DOMAIN,
    "conductor": DOMAIN,
    "environment": DOMAIN,
    "cylindrical_helix": DOMAIN,
    "numerics": DOMAIN,
    "runtime": DOMAIN,
    "ports": DOMAIN,
    # application
    "application": APPLICATION,
    # adapters (utility_functions and interfaces are transitional: they hold
    # both numerics and I/O; counted as adapters so that domain -> them is a
    # violation that the refactoring stages remove)
    "adapters": ADAPTERS,
    "utility_functions": ADAPTERS,
    "interfaces": ADAPTERS,
    # drivers
    "simulation": DRIVERS,
    "simulation_starter": DRIVERS,
    "opensc2_gui": DRIVERS,
    "drivers": DRIVERS,
}

# External top-level packages that must not be imported by a layer (and any
# layer further in).
FORBIDDEN_EXTERNAL: dict[int, frozenset[str]] = {
    DOMAIN: frozenset(
        {"pandas", "openpyxl", "yaml", "CoolProp", "matplotlib", "PIL", "tkinter",
         "IPython"}
    ),
    APPLICATION: frozenset(
        {"pandas", "openpyxl", "yaml", "CoolProp", "matplotlib", "PIL", "tkinter",
         "IPython"}
    ),
    ADAPTERS: frozenset({"tkinter"}),
    DRIVERS: frozenset(),
}

# --------------------------------------------------------------------------
# Baselines (today's violations; may only shrink)
# --------------------------------------------------------------------------

BASELINE_LAYER_VIOLATIONS: frozenset[str] = frozenset({
    "conductor.conductor_mesh -> pandas",
    "conductor.coupling -> pandas",
    "conductor.initialization_functions -> pandas",
    "conductor.input_loader -> adapters",
    "electromagnetics.circuit_topology -> pandas",
    "properties_of_materials.niobium_titanium -> pandas",
})

BASELINE_EXTERNAL_CONDUCTOR_ATTRIBUTES: frozenset[str] = frozenset({
    "adapters.output.checkpoint: cond_el_num_step",
    "adapters.output.checkpoint: cond_num_step",
    "adapters.output.checkpoint: cond_time",
    "adapters.output.checkpoint: electric_solution",
    "adapters.output.checkpoint: electric_time",
    "adapters.output.checkpoint: enthalpy_balance",
    "adapters.output.checkpoint: enthalpy_inl",
    "adapters.output.checkpoint: enthalpy_out",
    "adapters.output.checkpoint: i_save",
    "adapters.output.checkpoint: local_truncation_error_ratio",
    "adapters.output.checkpoint: previous_time_step",
    "adapters.output.checkpoint: restart_after_finalization",
    "adapters.output.checkpoint: time_step",
    "adapters.output.solution_writer: i_save",
    "electromagnetics.boundary_conditions: electric_known_term_vector",
    "electromagnetics.boundary_conditions: electric_reduction_operator",
    "electromagnetics.boundary_conditions: electric_retained_index",
    "electromagnetics.boundary_conditions: electric_stiffness_matrix",
    "electromagnetics.boundary_conditions: fixed_potential_index",
    "electromagnetics.boundary_conditions: fixed_potential_value",
    "electromagnetics.circuit_topology: _contact_nodes_first",
    "electromagnetics.circuit_topology: contact_incidence_matrix",
    "electromagnetics.circuit_topology: contact_nodes_current_carriers",
    "electromagnetics.circuit_topology: incidence_matrix",
    "electromagnetics.circuit_topology: incidence_matrix_transposed",
    "electromagnetics.conductance: electric_conductance_diag_matrix",
    "electromagnetics.conductance: electric_conductance_matrix",
    "electromagnetics.electric_solver: _steady_newton_iterations",
    "electromagnetics.electric_solver: _steady_state_permutation",
    "electromagnetics.electric_solver: cond_el_num_step",
    "electromagnetics.electric_solver: current_across",
    "electromagnetics.electric_solver: delta_voltage_across",
    "electromagnetics.electric_solver: electric_known_term_vector",
    "electromagnetics.electric_solver: electric_known_term_vector_old",
    "electromagnetics.electric_solver: electric_right_hand_side",
    "electromagnetics.electric_solver: electric_solution",
    "electromagnetics.electric_solver: electric_solution_steady",
    "electromagnetics.electric_solver: electric_stiffness_matrix",
    "electromagnetics.electric_solver: electric_time",
    "electromagnetics.electric_solver: nodal_potential",
    "electromagnetics.inductance: inductance_matrix",
    "electromagnetics.resistance: electric_differential_resistance_matrix",
    "electromagnetics.resistance: electric_resistance_matrix",
    "electromagnetics.resistance: parallel_jacket_pairs",
    "electromagnetics.system_assembly: electric_mass_matrix",
    "electromagnetics.system_assembly: electric_right_hand_side",
    "electromagnetics.system_assembly: electric_stiffness_matrix",
    "numerics.initial_flow: MDTINL",
    "numerics.transient_solution_functions: equation_eigenvalues",
    "numerics.transient_solution_functions: local_truncation_error_ratio",
    "numerics.transient_solution_functions: previous_time_step",
    "numerics.transient_solution_functions: time_accuracy_eigenvalue",
    "numerics.transient_solution_functions: time_step",
    "simulation: cond_num_step",
    "simulation: enthalpy_balance",
    "simulation: enthalpy_inl",
    "simulation: enthalpy_out",
    "simulation: hydraulic_formulation",
    "simulation: time_step",
    "thermal.energy_balance: E_jk_fin",
    "thermal.energy_balance: E_sol_fin",
    "thermal.energy_balance: E_str_fin",
    "thermal.energy_balance: energy_balance",
    "thermal.energy_balance: inner_pow",
    "thermal.energy_balance: mass_balance",
    "thermal.energy_balance: outer_pow",
})

BASELINE_DEFERRED_IMPORTS: frozenset[str] = frozenset({
    "components.jacket.jacket_component -> properties_of_materials.electrical_conductivity",
    "components.solid.solid_component -> electromagnetics.operating_conditions",
    "components.solid.strand_component -> properties_of_materials.electrical_conductivity",
    "conductor.conductor -> electromagnetics.electromagnetic_flags",
    "conductor.conductor -> electromagnetics.operating_conditions",
    "electromagnetics.electric_solver -> electromagnetics.electromagnetic_flags",
    "hydraulics.coolant -> hydraulics.hydraulic_flags",
})

CONDUCTOR_NAMES = frozenset({"conductor", "cond", "cond_r", "cond_c", "conductor_instance"})

# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def iter_source_files():
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or ".egg-info" in str(path):
            continue
        yield path


def parse(path: Path) -> ast.AST:
    # Some legacy modules contain invalid escape sequences in docstrings;
    # the resulting DeprecationWarnings are not this test's concern.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(path.read_text(encoding="utf-8"))


def module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "__init__"


def top_package(dotted: str) -> str:
    return dotted.split(".")[0]


def internal_packages() -> set[str]:
    names = set()
    for entry in SOURCE_ROOT.iterdir():
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.is_dir() and any(entry.glob("*.py")):
            names.add(entry.name)
        elif entry.suffix == ".py":
            names.add(entry.stem)
    return names


def resolve_import(node: ast.AST, module: str) -> list[str]:
    """Dotted names imported by an Import/ImportFrom node (absolute)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = module.split(".")
            base = base[: len(base) - node.level] if node.level <= len(base) else []
            prefix = ".".join(base)
            if node.module:
                prefix = f"{prefix}.{node.module}" if prefix else node.module
            return [prefix] if prefix else []
        return [node.module] if node.module else []
    return []


def collect_imports(path: Path):
    """Yield (target, is_function_local) for every import in the file."""
    tree = parse(path)
    module = module_name(path)

    def walk(node: ast.AST, local: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for target in resolve_import(child, module):
                    yield target, local
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from walk(child, True)
            else:
                yield from walk(child, local)

    yield from walk(tree, False)


def layer_violations() -> set[str]:
    internal = internal_packages()
    violations = set()
    for path in iter_source_files():
        module = module_name(path)
        source_layer = LAYER_OF.get(top_package(module))
        if source_layer is None:
            continue
        for target, _local in collect_imports(path):
            top = top_package(target)
            if top == "source_code":
                target = target[len("source_code."):]
                top = top_package(target)
            if top in internal:
                target_layer = LAYER_OF.get(top)
                if target_layer is not None and target_layer > source_layer:
                    violations.add(f"{module} -> {top}")
            elif top in FORBIDDEN_EXTERNAL[source_layer]:
                violations.add(f"{module} -> {top}")
    return violations


def deferred_imports() -> set[str]:
    """Function-local imports of internal modules, outside the drivers layer
    (drivers legitimately delay imports to control process setup, e.g. the
    matplotlib backend before anything imports pyplot)."""
    internal = internal_packages()
    found = set()
    for path in iter_source_files():
        module = module_name(path)
        if LAYER_OF.get(top_package(module)) == DRIVERS:
            continue
        for target, local in collect_imports(path):
            if local and top_package(target) in internal:
                found.add(f"{module} -> {target}")
    return found


def external_conductor_attributes() -> set[str]:
    found = set()
    for path in iter_source_files():
        module = module_name(path)
        if top_package(module) == "conductor":
            continue
        tree = parse(path)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for element in ast.walk(target):
                    if (
                        isinstance(element, ast.Attribute)
                        and isinstance(element.value, ast.Name)
                        and element.value.id in CONDUCTOR_NAMES
                        and isinstance(element.ctx, ast.Store)
                    ):
                        found.add(f"{module}: {element.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in CONDUCTOR_NAMES
            ):
                found.add(f"{module}: setattr(...)")
    return found


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def _assert_ratchet(current: set[str], baseline: frozenset[str], what: str):
    new = sorted(current - baseline)
    stale = sorted(baseline - current)
    message = []
    if new:
        message.append(f"new {what} (fix them or, if unavoidable, add to the baseline):")
        message.extend(f"  {entry}" for entry in new)
    if stale:
        message.append(f"stale baseline entries for {what} (delete them):")
        message.extend(f"  {entry}" for entry in stale)
    assert not message, "\n".join(message)


def test_no_new_layer_violations():
    _assert_ratchet(layer_violations(), BASELINE_LAYER_VIOLATIONS, "layer violations")


def test_no_new_external_conductor_attribute_assignments():
    _assert_ratchet(
        external_conductor_attributes(),
        BASELINE_EXTERNAL_CONDUCTOR_ATTRIBUTES,
        "conductor attributes assigned outside conductor/",
    )


def test_no_new_deferred_imports():
    _assert_ratchet(deferred_imports(), BASELINE_DEFERRED_IMPORTS, "function-local imports")


def test_every_top_level_package_has_a_layer():
    unassigned = sorted(
        internal_packages() - set(LAYER_OF) - {"gui_icons", "input_files", "__init__"}
    )
    assert not unassigned, f"assign a layer to: {unassigned}"


def _print_set(name: str, values: set[str]) -> None:
    print(f"{name}: frozenset[str] = frozenset({{")
    for value in sorted(values):
        print(f'    "{value}",')
    print("})\n")


if __name__ == "__main__":
    _print_set("BASELINE_LAYER_VIOLATIONS", layer_violations())
    _print_set("BASELINE_EXTERNAL_CONDUCTOR_ATTRIBUTES", external_conductor_attributes())
    _print_set("BASELINE_DEFERRED_IMPORTS", deferred_imports())
    sys.exit(0)
