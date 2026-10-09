# OPENSC2 architecture

*State after the clean-architecture refactoring (branch `clean-architecture`,
2026-10). The staged plan and its rationale are in
[architecture_refactoring_plan.md](architecture_refactoring_plan.md).*

OPENSC2 is organised in concentric layers whose dependencies point inward
only. All code lives in the `opensc2` package at the repository root; each
of its sub-packages (`opensc2.conductor`, `opensc2.adapters`, ...) is
assigned to one layer, so the layer of a module is the layer of its
sub-package. The entry points are `python -m opensc2.drivers.cli <model_dir>`
(or the installed `opensc2` command) and `python -m opensc2.simulation_starter`
(GUI).
`tests/test_architecture.py` checks the import direction on every test run
(baselines may only shrink); `lint-imports` (import-linter, dev dependency)
enforces the same contracts from `pyproject.toml`.

## Layers

| Layer | Packages | May import | External libraries |
|---|---|---|---|
| domain | `physical_fields`, `properties_of_materials`, `hydraulics`, `thermal`, `electromagnetics`, `components`, `conductor`, `environment`, `numerics`, `runtime`, `ports` | domain | numpy, scipy, numba |
| application | `application` | domain | numpy |
| adapters | `adapters` (`input`, `output`, `plotting`, `fluid_properties`) | application, domain | pandas, openpyxl, yaml, CoolProp, matplotlib |
| drivers | `simulation`, `drivers` (`cli`, `gui`), `simulation_starter` | everything | tkinter, PIL |

### Domain

Physics and numerics. `conductor.Conductor` is built from a typed
`ConductorDefinition` and owns typed state objects (`conductor.electric`,
`conductor.time_state`, `conductor.balances`, `conductor.save_schedule`);
physics functions keep the `(conductor, ...)` convention and read the
`runtime.SimulationContext` where they used to read the `Simulation` object.
Components are built from input/operations dataclasses and
`ComponentBuildSettings`; they look up materials in
`properties_of_materials.material.MATERIALS`. The domain never reads or
writes files: it depends on the Protocols in `ports/`.

### Ports (defined inward, implemented outward)

| Port | Used by | Implementation |
|---|---|---|
| `ports.fluid_properties.FluidPropertyProvider` | Coolant, FluidComponent, HydraulicNetwork, Environment | `adapters.fluid_properties.coolprop_provider` |
| `ports.input_source.InputSource` | input loaders, Simulation facade | `adapters.input.yaml_source`, `adapters.input.excel_source` (frozen legacy) |
| `ports.external_inputs.ExternalInputReader` | components, Conductor (tabulated series, scaling parameters, flow BCs, user grids) | `adapters.input.external_inputs` |
| `ports.observer.SimulationObserver` | application use cases | `adapters.output.observers`, `adapters.plotting.observers` |
| `ports.diagnostics.DiagnosticsSink` | numerics (non-finite state dumps) | `adapters.output.diagnostics_npz` |

### Application

`application.build_simulation` (instantiate and initialize the conductors),
`application.transient_loop.run_transient` (the time loop),
`application.post_processing.post_process`. They operate on a
`runtime.SimulationContext` and notify observers at fixed points; they know
nothing about files or plots.

### Composition root

`simulation.Simulation(base_path, step_callback=None).run()` reads the deck
through `adapters.input.open_input_source`, builds the context, registers
the adapters behind the ports and delegates to the application. Its legacy
attributes (`transient_input`, `list_of_Conductors`, `dict_path`, ...) proxy
to the context, so drivers' step callbacks keep working, including writes to
`transient_input`.

## Recipes

* **New output or plot**: implement `ports.observer.SimulationObserver`
  (subclass `NullObserver`) in `adapters/`, register it in
  `Simulation.run()`. Never write files from the domain.
* **New input format**: implement `ports.input_source.InputSource` in
  `adapters/input/`, pick it in `adapters.input.open_input_source`. The
  loaders (`adapters/input/component_loaders.py`,
  `conductor_loader.py`) are format-agnostic.
* **New material**: add the correlations module under
  `properties_of_materials/`, register a `Material` in
  `properties_of_materials/material.py` (choose the signature kind), add an
  alias if the deck uses another name.
* **New superconductor**: register a `Superconductor` instead (its Jc, Tc
  and Tcs correlations plus the `critical_signature` giving their argument
  order; leave Tc out for a constant Tc0). `StrandComponent` binds it from
  `inputs.superconducting_material` at construction; an unknown name fails
  there.
* **New fluid property library**: implement `FluidPropertyProvider`, register
  a factory with `ports.fluid_properties.register_provider_factory`.
* **New external quantity read from file**: add a method to
  `ExternalInputReader` and its file adapter; call it from the domain.

## Verification of refactoring stages

`tests/tools/compare_trees.py record|check CASE_1|CASE_2|CASE_3` hashes every
file of a run's Output tree; refactoring commits must keep the trees
byte-identical (stricter than the 1e-5 regression tolerance).
`pytest -m "not slow"` is the developer loop; the full suite includes the
regression, checkpoint-restart SHA and verification tests.

## Known physics caveats (recorded, not changed by the refactoring)

* Transverse insulation coupling evaluates both temperatures at
  source-build time. Network-coupled conductors are all prepared before the
  shared solve of a step, so every patch pair is exactly antisymmetric. A
  conductor without network ports is stepped inside the same loop iteration
  (`application/transient_loop.py`), so a later conductor in the list reads
  its partner's already-updated field; the exchange is then not exactly
  antisymmetric for such pairs.

## Remaining debt (ratchet baselines)

* `models/` driver scripts are not tracked on this branch; their imports
  were updated in place together with `tests/`, `scripts/` and `tools/`.
