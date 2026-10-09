"""Architecture ratchet: layered dependencies of the ``opensc2`` package.

Target architecture (docs/architecture_refactoring_plan.md): concentric layers
whose dependencies point inward only.

    0 domain       physics, numerics, typed inputs, ports (Protocols)
    1 application  use cases (initialization, time loop, restart, observers)
    2 adapters     input readers, output writers, plotting, property libraries
    3 drivers      Simulation facade, GUI, CLI

Because every subfolder of ``opensc2`` is a sub-package assigned to a layer, a top
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
  (the way import cycles are currently hidden), as ``"module -> target"``;
* ``BASELINE_LEGACY_CONDUCTOR_READS``: reads of a conductor state attribute
  by its former loose name (``conductor.cond_time`` instead of
  ``conductor.time_state.cond_time``, ``getattr(conductor, "nodal_potential")``
  instead of ``conductor.electric.nodal_potential``), as ``"module: attribute"``.

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
PACKAGE = "opensc2"
SOURCE_ROOT = REPOSITORY_ROOT / PACKAGE
# ``python tests/test_architecture.py`` (baseline printing) runs without the
# conftest: make the package importable from the checkout.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
    # adapters
    "adapters": ADAPTERS,
    # drivers
    "simulation": DRIVERS,
    "simulation_starter": DRIVERS,
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

BASELINE_LAYER_VIOLATIONS: frozenset[str] = frozenset()

BASELINE_EXTERNAL_CONDUCTOR_ATTRIBUTES: frozenset[str] = frozenset({
})

BASELINE_DEFERRED_IMPORTS: frozenset[str] = frozenset()

BASELINE_LEGACY_CONDUCTOR_READS: frozenset[str] = frozenset()

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


def strip_package(dotted: str) -> str:
    """``opensc2.conductor.x`` -> ``conductor.x`` (package-relative name)."""
    if dotted == PACKAGE:
        return ""
    if dotted.startswith(PACKAGE + "."):
        return dotted[len(PACKAGE) + 1:]
    return dotted


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
            target = strip_package(target)
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
            target = strip_package(target)
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


def legacy_conductor_reads() -> set[str]:
    """``module: attribute`` for every access to a conductor state attribute by
    its former loose name (see conductor.electric_state.ELECTRIC_ATTRIBUTES and
    conductor.run_state.RUN_STATE_ATTRIBUTES), as an attribute of a conductor
    variable or through getattr/hasattr/setattr with a literal name."""
    from opensc2.conductor.electric_state import ELECTRIC_ATTRIBUTES
    from opensc2.conductor.run_state import RUN_STATE_ATTRIBUTES

    legacy = set(ELECTRIC_ATTRIBUTES) | set(RUN_STATE_ATTRIBUTES)
    found = set()
    for path in iter_source_files():
        module = module_name(path)
        for node in ast.walk(parse(path)):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in legacy
                and isinstance(node.value, ast.Name)
                and node.value.id in CONDUCTOR_NAMES
            ):
                found.add(f"{module}: {node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr", "setattr")
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in CONDUCTOR_NAMES
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in legacy
            ):
                found.add(f"{module}: {node.func.id}({node.args[1].value!r})")
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


def test_no_legacy_conductor_state_reads():
    _assert_ratchet(
        legacy_conductor_reads(),
        BASELINE_LEGACY_CONDUCTOR_READS,
        "reads of conductor state by its former loose attribute name",
    )


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
    _print_set("BASELINE_LEGACY_CONDUCTOR_READS", legacy_conductor_reads())
    sys.exit(0)
