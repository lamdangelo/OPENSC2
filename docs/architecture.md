# OPENSC2 architecture

*State after the clean-architecture refactoring (branch `clean-architecture`,
2026-10). The staged plan and its rationale are in
[architecture_refactoring_plan.md](architecture_refactoring_plan.md).*

OPENSC2 is organised in concentric layers whose dependencies point inward
only. `pyproject.toml` maps every sub-folder of `source_code/` to a top-level
package, so the layer of a module is the layer of its top-level package.
`tests/test_architecture.py` checks the import direction on every test run
(baselines may only shrink); `lint-imports` (import-linter, dev dependency)
enforces the same contracts from `pyproject.toml`.

## Layers

| Layer | Packages | May import | External libraries |
|---|---|---|---|
| domain | `physical_fields`, `properties_of_materials`, `hydraulics`, `thermal`, `electromagnetics`, `components`, `conductor`, `environment`, `numerics`, `runtime`, `ports` | domain | numpy, scipy, numba |
| application | `application` | domain | numpy |
| adapters | `adapters` (`input`, `output`, `plotting`, `fluid_properties`) | application, domain | pandas, openpyxl, yaml, CoolProp, matplotlib |
| drivers | `simulation.py`, `drivers` (`cli`, `gui`), `simulation_starter.py` | everything | tkinter, PIL |

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

## Remaining debt (ratchet baselines)

* `conductor.conductor_mesh` and `electromagnetics.circuit_topology` build
  pandas DataFrames as in-memory tables (nodal coordinates, connectivity,
  contact nodes) read by `electromagnetics.inductance`. No I/O is involved;
  replacing them by numpy structures is a follow-up.
* `components.solid.solid_component` imports
  `electromagnetics.operating_conditions` inside a function: a genuine
  cycle (that module imports the strand classes).
* `SolidComponent.get_transverse_coupling` still parses its CSV patch file
  and walks `simulation.list_of_Conductors` lazily; moving this to the
  external-input reader and the application layer is the next step.
* The superconductor critical-property dispatch in
  `StrandComponent.eval_critical_properties` / `_critical_current_density`
  still branches on the material string (a `Superconductor` protocol would
  complete Stage 8).
* Test fakes (`SimpleNamespace` conductors, `SolidComponent.__new__`) still
  rely on the old attribute spelling that the Conductor forwards; the
  forwarding (`Conductor._FORWARDED`) can be removed once the tests build
  real objects (`tests/factories.py` is the intended home).
* `models/` driver scripts are not tracked on this branch; their imports
  were updated in place together with `tests/`, `scripts/` and `tools/`.
* The root-package move (`opensc2/...`) is deferred (appendix of the plan).
