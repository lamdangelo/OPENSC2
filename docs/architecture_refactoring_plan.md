# OPENSC2 Clean-Architecture Refactoring Plan

*Status: stages 0–9 implemented on branch `clean-architecture` (2026-10-08); see [architecture.md](architecture.md) for the resulting structure and the remaining debt. Scope: structure only — no numerical change in any stage.*

This document analyses the current dependency structure of `source_code/` and lays out a
staged plan to reach a layered ("clean") architecture in the sense of Robert C. Martin:
concentric layers whose dependencies point only inward, with physics and numerics in the
centre and files, plots, GUI and third-party libraries at the rim. It continues the
modularization documented in [opensc2_refactoring_summary.tex](opensc2_refactoring_summary.tex),
which created the physics packages but did not impose a dependency direction.

Decisions already taken:

* The root-package move (`opensc2/…`) is **deferred** (see Appendix); existing packages are
  assigned to layers in place.
* The Excel workbook input path is **kept but frozen**: one legacy adapter, no new features.

---

## 1. Findings

Measured with an AST pass over `source_code/` (all imports, including function-local ones)
and three code audits. Line numbers refer to the state of branch `w7x-model` at commit
`0029025`.

### 1.1 Dependency graph

Every physics package is in a cycle with every other one:

```
conductor ↔ components ↔ thermal ↔ electromagnetics ↔ hydraulics ↔ utility_functions
```

Module-level cycles are avoided only by five function-local imports
([conductor/conductor.py:399](../source_code/conductor/conductor.py#L399),
[conductor/conductor_mesh.py:864](../source_code/conductor/conductor_mesh.py#L864),
[hydraulics/network/coupling.py:592](../source_code/hydraulics/network/coupling.py#L592),
[utility_functions/plots.py:224](../source_code/utility_functions/plots.py#L224),
[utility_functions/checkpoint.py:286](../source_code/utility_functions/checkpoint.py#L286))
and a lazy `__getattr__` in [electromagnetics/__init__.py](../source_code/electromagnetics/__init__.py).
[components/component_factory.py:25](../source_code/components/component_factory.py#L25)
type-imports `simulation.Simulation`, i.e. the component layer refers to the outermost
orchestrator.

### 1.2 `utility_functions/` is a hub

It mixes the numerics core (`transient_solution_functions.py` 1307 lines,
`step_matrix_construction.py` 637, `gen_flow.py` 871) with all I/O (`output.py` 1400,
`plots.py` 1481, `checkpoint.py`, `simulation_log.py`, `simulation_paths.py`,
`auxiliary_functions.py` Excel/CSV readers). It imports every package and is imported by
every package.

### 1.3 Inner objects do outer work

* `Conductor.__init__(base_path, ICOND, magnet_file_name)` reads Excel/YAML itself
  ([conductor.py:128-144](../source_code/conductor/conductor.py#L128-L144), :209, :222-234,
  :406, :412, :445). `Conductor.initialization` writes files and updates matplotlib real-time
  plots (:845-864); `post_processing` writes files (:1173-1199); figure/canvas dictionaries
  live on the physics object (:472-475); `load_user_defined_quantity` dispatches through
  `simulation.func_open_aux` (:1291).
* Solid components read external current/field/heat workbooks lazily *inside physics
  methods* when `cond_time == 0`
  ([solid_component.py:398](../source_code/components/solid/solid_component.py#L398), :570,
  :697), parse CSV (:116-140) and iterate `simulation.list_of_Conductors` (:127-130).
  [strand_component.py:704](../source_code/components/solid/strand_component.py#L704) opens
  `scaling_input.dat` relative to the current working directory.
* Fluid boundary conditions from file re-read a workbook at **every assembly**
  ([fluid_component.py:274](../source_code/components/fluid/fluid_component.py#L274), :398, :506).
* `Environment(f_path)` reads its own file, keeps a raw dict (`inputs["Medium"]`) that physics
  reads directly (jacket_component.py:148, thermal/energy_balance.py:116,
  thermal/radiation.py:39), and calls CoolProp `PropsSI` directly
  ([environment.py:180-190](../source_code/environment/environment.py#L180-L190)), bypassing
  `interfaces/coolprop_interface.py`.
* The numerics core writes `.npz` dumps to the cwd
  ([transient_solution_functions.py:820](../source_code/utility_functions/transient_solution_functions.py#L820),
  :880) and prints every step (:165, :286). `plots.py` selects the matplotlib backend at
  import; `mass_flow_equations.py:87` reads an environment variable at import;
  `Simulation.CWD = os.getcwd()` is a class attribute.
* Core code depends on the orchestrator: `simulation.transient_input[...]` (47 sites),
  `simulation.dict_path` (69), `simulation.environment` (30),
  `simulation.fluid_prop_aliases` (thermal/htc_evaluation.py:36).

### 1.4 `Conductor` is a god object

More than 50 attributes are assigned from outside the class: `electromagnetics/*` sets about
25 `electric_*`/`incidence_*`/`contact_*` matrices; `transient_solution_functions` sets the
time-step state; `output.py` sets `i_save`; `thermal/energy_balance.py` sets `E_*_fin`;
`checkpoint.py` restores with `setattr`. Topology is held in nested string-keyed dicts
(`dict_topology`, `dict_interf_peri["ch_ch"]["Open"]["Gauss"]`, used in 7 files) plus a
namedtuple built in [conductor_topology.py:1178-1290](../source_code/conductor/conductor_topology.py#L1178-L1290).

### 1.5 Materials and enums

Material dispatch is lowercase string → module-level dict of functions with
*heterogeneous signatures*, so callers branch on `func.__name__`
([stack_component.py:458](../source_code/components/solid/stack_component.py#L458), :485;
[strand_mixed_component.py:384-398](../source_code/components/solid/strand_mixed_component.py#L384-L398),
:427-438). Superconductor critical properties are duplicated if/elif chains
(strand_component.py:206-250, :457-530); the vocabulary is inconsistent (`re123` vs `ybco`).
Enums carry their file-code translators; `CouplingMatrix` stores raw codes, so physics
decodes `HTC_Choice.get_htc_choice_flag(...)` at 14 sites in `thermal/htc_evaluation.py`
and `ElectricConductanceMode` in `electromagnetics/conductance.py:106`,
`resistance.py:121/178`. Physics mutates operations dataclasses after construction
(`deal_with_flag_IOP_MODE`, solid_component.py:1167-1198). Some loaders pass `str` or raw
`int` where enums are typed (jacket_component_inputs.py:105-136).

### 1.6 Clean islands (templates to copy)

`physical_fields/` (FieldContainer); `hydraulics/network/` (HydraulicNetwork takes a
validated dataclass, `from_mapping` + `validate`); friction/Nusselt strategy classes and
their factories; `properties_of_materials/*` (mostly pure functions); per-component matrix
builders (`build_smat_fluid_energy(matrix, f_comp, eq_idx)`, `conduction.build_mmat_solid`);
the solver dataclasses in `conductor/solver_structures.py`; the `ConstantFluidProperties`
test double; the `step_callback` and `fluid_source_callback` hooks.

### 1.7 Symptoms

Unit tests build components with `SolidComponent.__new__` or `SimpleNamespace` fakes
(tests/test_coupling_loss.py:28, tests/test_mass_flow_explicit.py:139-227). Bugs hide at the
seams: the GUI calls `conductor_initialization(self)` / `conductor_solution(self)` with a stray
argument and never creates `simulation_summary` (opensc2_gui.py:473/475);
components/component_flags.py:28 and component_collection.py:113 build a `ValueError` and
never raise it; conductor/input_loader.py:234 passes a literal `Ellipsis`;
environment.py:524/545 call an undefined `eval_air_properties`.

### 1.8 Extension points that must keep working

`Simulation(base_path, step_callback=None).run()` is used by about 25 scripts under
`models/` and by `tests/verification/case_builders.py`. Callbacks write
`simulation.transient_input["STPMAX"/"TEND"]`, `strand.operations.*`,
`strand.current_interpolator`, `conductor.fluid_source_callback`, and read
`simulation.list_of_Conductors` and `simulation.hydraulic_network`.

---

## 2. Target architecture

`pyproject.toml` sets `package-dir = {"" = "source_code"}`, so every subfolder is a
top-level import. Existing physics packages are therefore *assigned* to a layer in place
(no moves, history preserved); only misplaced modules move into **new** sibling packages.
Never use stdlib-clashing package names (`io`, `logging`, `types`).

```
source_code/
  ── DOMAIN (entities, pure kernels, domain services) ────────────────────────
  physical_fields/         unchanged
  properties_of_materials/ unchanged; later + material.py (Protocol + registry)
  hydraulics/  thermal/  electromagnetics/  components/  conductor/  environment/
                           unchanged locations; loaders and I/O leave them
  numerics/        NEW     transient_solution_functions, step_matrix_construction,
                           time_step (get_time_step), initial_flow (gen_flow physics),
                           interpolation (pure helpers from auxiliary_functions)
  runtime/         NEW     SimulationContext, SimulationClock, TimeSteppingSettings
                           (plain dataclasses: what physics needs from "simulation")
  ports/           NEW     Protocols: FluidPropertyProvider, InputSource,
                           SimulationObserver, DiagnosticsSink
  ── APPLICATION (use cases) ──────────────────────────────────────────────────
  application/     NEW     build_simulation (initialization), transient_loop (the
                           while loop), restart, post_processing, observers
                           (CompositeObserver, CallbackObserver wrapping step_callback)
  ── ADAPTERS ─────────────────────────────────────────────────────────────────
  adapters/        NEW
    input/                 yaml_source, excel_source (legacy, frozen), yaml_schema_v2,
                           conductor_loader, component_loaders, network_loader,
                           environment_loader, auxiliary_files, external_inputs, codes
    output/                solution_writer (ex output.py), checkpoint, simulation_log,
                           simulation_paths, diagnostics_npz
    plotting/              plots, real_time_plots
    fluid_properties/      coolprop_provider, constant_provider (ex interfaces/)
  ── DRIVERS ──────────────────────────────────────────────────────────────────
  simulation.py            facade `Simulation(base_path, step_callback=None).run()`
                           + composition root
  drivers/         NEW     cli.py, gui/ (ex opensc2_gui.py, gui_icons/)
  utility_functions/, interfaces/   transitional re-export shims, deleted last
```

### 2.1 Layer rules

| Layer | May import | External libraries allowed | Forbidden |
|---|---|---|---|
| domain: physical_fields, properties_of_materials, hydraulics, thermal, electromagnetics, components, conductor, environment, numerics, runtime, ports | domain only (siblings may import each other; no function-local imports by the end) | numpy, scipy, numba, stdlib | pandas, openpyxl, yaml, CoolProp, matplotlib, PIL, tkinter, `print`, `os.getcwd`, file writes |
| application | domain | numpy, logging | all of the above, concrete adapters |
| adapters | application, domain | pandas, openpyxl, yaml, CoolProp, matplotlib | tkinter |
| drivers: simulation.py, drivers/ | everything | everything | — |

### 2.2 Dependency inversions (ports defined inward, implemented outward)

1. **`FluidPropertyProvider`** (ports/) — exactly the methods `coolant.py`, `channel.py`,
   `hydraulic_network.py` and `fluid_component.py` call today. Implemented by
   `adapters/fluid_properties/coolprop_provider` (including the tabulated splines) and
   `constant_provider`; injected through constructors; chosen from `FluidType` in the
   composition root. `fluid_prop_aliases` becomes a provider attribute.
2. **`InputSource`** (ports/) — `transient_settings()`, `environment()`,
   `conductor_definition(i)`, `table(file_key, sheet)`, `auxiliary_series(...)`,
   `hydraulic_network()`. Implemented by `yaml_source` and the frozen `excel_source`.
   Produces typed definitions: `ConductorDefinition{inputs, operations, files, grid,
   coupling, components: [ComponentDefinition], external series}`. External profiles are
   preloaded into interpolator objects (pattern: the already sanctioned
   `strand.current_interpolator`).
3. **`SimulationObserver`** (ports/) — `on_initialized`, `before_step` (truthy return
   stops, i.e. the `step_callback` contract), `after_conductor_step`, `after_step`,
   `on_autosave`, `on_finished`. Writers, plots, summary, autosave and console progress
   become observers.
4. **`DiagnosticsSink`** — the `.npz` failure dumps become an injected callable (no-op
   default; the facade wires an npz writer to the same location as today).
5. **`Material` / `Superconductor`** Protocols and `MATERIAL_REGISTRY`
   (properties_of_materials/) — thin wrapper classes normalize the heterogeneous
   signatures while calling the *same* functions.
6. **`SimulationContext`** (runtime/) replaces the `simulation` parameter of physics
   functions: `transient_settings` (**the same mutable dict object** the facade exposes as
   `transient_input`, so driver writes keep working), `conductors`, `environment`,
   `fluid_properties`, `hydraulic_network`, `qsources`, `clock`, `time_stepping`.

### 2.3 Conductor decision

`Conductor` stays in the domain and physics keeps the `(conductor, …)` signature
established in the previous refactoring. The god object is broken by extracting typed
state sub-objects (`conductor.electric`, `conductor.time_integration`,
`conductor.balances`, `conductor.topology`) with transitional property aliases — not by
rewriting the physics functions.

---

## 3. Stages

Each stage is one mergeable PR; the code runs and all tests pass at the end of each.
Size legend: S < 300 lines changed, M 300–1000, L 1000–2500, XL > 2500 (mechanical moves).

### Stage 0 — Safety net and architecture ratchet (S)

* `tests/test_architecture.py` (pure `ast`, no new dependency): walks
  `source_code/**/*.py` including function-local imports, maps package → layer via a
  `LAYER_OF` table, and asserts (a) no inner→outer edge, (b) no forbidden external library
  per layer, (c) today's violations are in a `BASELINE` set. The test fails both on a new
  violation *and* on a stale baseline entry, so the baseline can only shrink. Two further
  ratchets: `(module, attr)` pairs where `conductor.attr = …` is assigned outside
  `conductor/` (about 55 today), and `DEFERRED_IMPORTS_ALLOWED` (the five cycle edges).
* pytest markers `slow`, `regression`, `verification` in `pyproject.toml`; mark CASE_1/2/3,
  the checkpoint-restart test and `tests/verification`. Developer loop: `pytest -m "not slow"`.
* `tests/tools/compare_trees.py`: run CASE_1 (YAML) into a scratch directory and compare the
  Output tree **bitwise** with a baseline recorded on `main` (reuse the hashing in
  [tests/test_checkpoint_restart.py:173](../tests/test_checkpoint_restart.py#L173)). This is
  the gate for every later PR and is stricter than the 1e-5 regression tolerance.

### Stage 1 — Cheap boundary fixes, no file moves (S–M)

* Matplotlib backend selection moves out of `plots.py` import time into the facade/GUI.
  The environment-variable read in `mass_flow_equations.py:87` becomes a lazy getter.
  `Simulation.CWD` becomes an instance attribute.
* The `.npz` dumps in `finalize_step` and `coolprop_interface` go through a
  `DiagnosticsSink` argument (no-op default; npz adapter wired by the facade). The
  `get_time_step` prints become a `report` keyword (replaced by an observer in Stage 5).
  The hard-coded `IADAPTIME == 3` schedule stays untouched (numerics).
* Hygiene with zero numerical effect: GUI stray-argument calls → `simulation.run()`;
  missing `raise` in `component_flags.py:28` / `component_collection.py:113`; delete dead
  `utility_functions/electric_auxiliary_functions.py` and `save_ndarray` (hard-coded `D:/`
  path); remove stray prints in `properties_of_materials`. Do **not** touch the in-place
  array mutation at `niobium_titanium.py:444` / `magnesium_diboride.py:310` yet.
* Add `python -m drivers.cli <model_dir>` (thin) so headless use needs no `sys.path` hacks.

### Stage 2 — Split `utility_functions` into `numerics/` and `adapters/`; introduce import-linter (XL, mechanical)

* `git mv` with re-export shims under the old paths (emitting `DeprecationWarning`) so
  `models/` and `tests/` keep importing.
  * `numerics/`: `transient_solution_functions`, `step_matrix_construction`, interpolation
    helpers (`build_interpolator`, …) from `auxiliary_functions`.
  * `conductor/`: `initialization_functions` (then `conductor_mesh.py:864` becomes a
    module-level import).
  * `numerics/initial_flow`: `gen_flow` physics; its `get_from_xlsx` call sites become a
    `flow_parameters` callable argument (default passed by the caller; baseline entry until
    Stage 4).
  * `adapters/input/auxiliary_files`: `with_read_csv`, `with_read_excel`, `get_from_xlsx`,
    `load_auxiliary_files`.
  * `adapters/output/`: `output`, `checkpoint`, `simulation_log`, `simulation_paths`;
    `adapters/plotting/`: `plots`.
  * `interfaces/` → `adapters/input/yaml_input_registry` + `yaml_schema_v2`,
    `adapters/fluid_properties/coolprop_provider`.
* Resolve the function-local cycle imports that the move makes unnecessary
  (`conductor.py:399`, `conductor_mesh.py:864`, `coupling.py:592`, `plots.py:224`,
  `checkpoint.py:286`). `thermal/energy_equation.py:43` imports `numerics` (no cycle once
  numerics no longer imports `conductor.conductor`; type-only uses go under `TYPE_CHECKING`).
* `pyproject.toml`: `import-linter>=2.0` in `dev`; `[tool.importlinter]` with
  (1) a `layers` contract
  `simulation|drivers / adapters / application / numerics:runtime:ports:conductor:components:thermal:electromagnetics:hydraulics:environment:properties_of_materials:physical_fields`
  whose `ignore_imports` mirrors the pytest baseline, (2) a `forbidden` contract from domain
  and application to pandas, openpyxl, yaml, CoolProp, matplotlib, PIL, tkinter, IPython,
  (3) a `forbidden` contract from adapters to tkinter. ruff rule `T20` (no print) with
  per-file ignores for adapters, drivers, `simulation.py`, tests. CI runs `lint-imports`.

### Stage 3 — Fluid-property port and Environment (M)

* `ports/fluid_properties.py` Protocol; `coolprop_provider` implements it;
  `ConstantFluidProperties` becomes an injected provider for `FluidType.CONSTANT` (keep
  `set_constant_fluid_properties` as a shim so `tests/verification/conftest.py` and
  `case_builders.py` keep working, then migrate them).
* Inject the provider into `Coolant`, `FluidComponent`, `HydraulicNetwork`, `Environment`;
  `htc_evaluation.py:36` reads `provider.aliases`.
* `environment/`: `EnvironmentState` dataclass with typed fields (replaces the
  `inputs["Medium"]` reads at jacket_component.py:148, energy_balance.py:116,
  radiation.py:39) plus air-property evaluation through the provider; the file read moves
  to `adapters/input/environment_loader`. Fix the enum-vs-string dict keys (:149-160) and
  the undefined `eval_air_properties` only after a unit test proves those branches are
  unreachable with current inputs.

### Stage 4 — Input layer extraction (three PRs)

**4a — `InputSource` port and the two sources (M).** `ports/input_source.py`;
`adapters/input/yaml_source` (wraps `yaml_input_registry`; the module-global
`_registry_cache` becomes instance state) and `excel_source` (frozen legacy; all
`pd.read_excel` / `load_workbook`). The ~10 `if get_registry(...) … else pd.read_excel`
sites collapse into the composition root. The loader halves of `components/**/*_inputs.py`
and `conductor/input_loader.py`, `input_validator.py` move to `adapters/input/` (the
dataclasses stay; shims are left). This also frees `hydraulics/channel.py`, `coolant.py`,
`friction_factor_factory.py` and `thermal/heat_transfer_factory.py` from pandas/yaml.

**4b — Constructors from definitions (L).** `conductor/conductor_definition.py`
(`ConductorDefinition`, `ComponentDefinition`); `Conductor.__init__(definition, counter)`;
`adapters/input/conductor_loader.load_conductor(base_path, i, source)` reproduces the
legacy path (one-stage positional shim on the old signature). Components drop the
`simulation` and `conductor` constructor parameters (`IADAPTIME`/`STPMIN` →
`runtime.TimeSteppingSettings`; consistency checks take `zlength`). `SolidComponent`
inherits `Component`. `_build_transverse_patches` CSV parsing → `adapters/input/external_inputs`;
the cross-conductor wiring → `application/build_simulation`. `ComponentFactory` loses
`ComponentBuildContext(simulation, conductor)` and the `TYPE_CHECKING` import. Replace the
`SolidComponent.__new__` / `SimpleNamespace` fakes in `tests/test_coupling_loss.py`,
`test_induced_losses_relaxation_and_stack.py`, `test_mass_flow_explicit.py` with real
constructors via a new `tests/factories.py`.

**4c — External series and code decoding (M–L).** Preload all lazy reads
(solid_component.py:398/570/697, strand_component.py:382/655/704,
fluid_component.py:274/398/506, conductor.py:412/445/1291, the user-defined grid at
conductor_mesh.py:864) into interpolator objects (`numerics/interpolation.ExternalSeries`)
at build time, using the *same* interpolation functions on the *same* arrays (unit test:
load one `TDD_examples` file both ways and assert array equality). `scaling_input.dat` is
resolved relative to `base_path` (verify by grep that every driver and test runs with the
model directory as cwd). Decode raw codes once in the loader (`HTC_Choice`,
`ElectricConductanceMode`, `CurrentMode` → `adapters/input/codes.py`); `CouplingMatrix`
stores enums; the 14 + 3 per-call decodes become plain comparisons. Normalize
`deal_with_flag_IOP_MODE` and `magnetic_field_units = None` into the loader; fix the
str/int-versus-enum loader inconsistencies; operations dataclasses stay mutable (drivers
write to them).

### Stage 5 — Output and observer extraction (L)

* `ports/observer.py` `SimulationObserver`; `application/observers.py` `CompositeObserver`,
  `CallbackObserver(step_callback)`.
* Adapters wrapping the existing function bodies one-to-one: `SpatialDistributionWriter`
  (`i_save` stays on the conductor in this stage because checkpoint restores it),
  `TimeEvolutionWriter` (including the final flush and the autosave `flush_only`),
  `NetworkWriter`, `SummaryObserver`, `RealTimePlotObserver` (owns the figure/canvas dicts
  from conductor.py:472-475), `CheckpointAutosaver`, `ConsoleProgress`.
* `Conductor.initialization` loses its save/plot calls (:845-864) → `on_initialized`;
  `Conductor.post_processing` → `application/post_processing` + `SolutionWriter`.
  `simulation.dict_path` → an `OutputPaths` object (`adapters/output/simulation_paths`) with
  a read-only mapping view for one stage.
* **Ordering guard.** The exact call order of `simulation.py:389-648` is tabulated in
  `transient_loop.py` and preserved: radiative → save at t = 0 → loop { callback →
  get_time_step → lockstep → clock → print → electric / operating_conditions → heat source →
  step or coupled step → density refresh and enthalpy → radiative → plots → save_space →
  summary → save_time → save_network → autosave } → final flush. The SHA test in
  `tests/test_checkpoint_restart.py` is the sharp guard.
* Checkpoint write/restore goes through `conductor.snapshot()` / `restore(snapshot)`
  dataclasses instead of `setattr`; the on-disk npz format is unchanged.

### Stage 6 — Application layer, `SimulationContext`, thin facade (L)

* `runtime/context.py` `SimulationContext` (see 2.2); `application/transient_loop.py`
  `run_transient(ctx, observer)` is the body of `conductor_solution`;
  `application/build_simulation.py` is `conductor_instance` + `conductor_initialization` +
  network build; `application/restart.py`, `post_processing.py`.
* Physics signatures narrow from `simulation` to `ctx` (same object, attribute renames
  only): conductor.py:427/430/594/709-725, `operating_conditions_th`, `build_heat_source`,
  `htc_evaluation` (:571 environment), `energy_balance`, `radiation`, `network/coupling`,
  `conductor_mesh`.
* `simulation.py` facade: `transient_input`, `environment`, `list_of_Conductors`,
  `hydraulic_network`, `simulation_time`, `num_step`, `dict_path`, `simulation_summary`
  become properties onto the context and adapters (read and write where drivers write);
  `run()` = initialize → [restart] → run_transient → post_process → summary.write;
  `conductor_initialization()` / `conductor_solution()` / `conductor_post_processing()` stay
  as thin delegates for the GUI.
* Remove the remaining `ignore_imports` entries for `conductor/`, `thermal/`, `hydraulics/`,
  `components/`.

### Stage 7 — Conductor state extraction (3 × M)

* **7a** `conductor/electric_state.py::ElectricState` (the ~25 attributes assigned by
  `electromagnetics/*`); `Conductor.electric`; transitional getter/setter aliases for every
  old name; migrate the `electromagnetics/*` writes in the same PR; checkpoint serializes
  the dataclass.
* **7b** Extend `conductor/solver_structures.py::TimeIntegrationState` with `time_step`,
  `previous_time_step`, `local_truncation_error_ratio`, `time_accuracy_eigenvalue`,
  `equation_eigenvalues`, `cond_time`, `cond_num_step`, `restart_after_finalization`;
  add `BalanceState` (`E_*_fin`, `enthalpy_*`, `inner_pow`, `outer_pow`, `mass_balance`,
  `MDTINL`) and `SaveSchedule` (`i_save`, `num_step_save`, `n_digit_time`).
* **7c** `conductor/topology_state.py::TopologyState` wrapping `dict_topology`,
  `dict_interf_peri`, `interface`, `equation_index`; keep the nested-dict `__getitem__`
  shapes so the 13 consumers are untouched; add typed accessors; migrate consumers
  opportunistically.
* The attribute-assignment ratchet reaches zero; delete the aliases.

### Stage 8 — Material protocol and registry (M–L)

* `properties_of_materials/material.py`: `Material` and `Superconductor` Protocols;
  wrapper classes per module (`CopperNIST`, `NbTi`, `Nb3Sn`, `MgB2`, `RE123`,
  `StainlessSteel`, …) normalizing the signatures; `MATERIAL_REGISTRY` with aliases
  (`re123` ≡ `ybco`, `cu` ≡ `copper`); resolution in the loader → `inputs.material: Material`;
  merge the `properties_of_materials/electrical_conductivity.py` alias dispatcher.
* Replace the dispatch dicts and `func.__name__` branches
  (stack_component.py:76-118/458/485, strand_mixed_component.py:62-95/384-398/427-438,
  jacket_component.py:35-50, strand_component.py:206-250/457-530). Table-driven unit test:
  old dict entry vs new wrapper on a (T, B) grid, bitwise equal.
* In-place caller-array mutation (niobium_titanium.py:444, magnesium_diboride.py:310):
  first test whether any caller observes it; copy inputs only if none does.

### Stage 9 — Drivers layer and tightening (M)

* `drivers/cli.py`, `drivers/gui/` (`opensc2_gui.py`, `gui_icons`); composition helpers used
  by `simulation.py`. Update the ~25 `models/**/run_*.py` and test imports (mechanical sed);
  delete the `utility_functions/` and `interfaces/` shims; import-linter `ignore_imports`
  empty; architecture-test baselines empty and asserted empty.
* `docs/architecture.md` with the layer table and recipes: how to add an input format, an
  output, a material, an observer.

---

## 4. What not to do

* No change to any numerical algorithm, call order (momentum before energy on the shared
  matrix), time-step schedule (including `IADAPTIME == 3`), tolerance, interpolation
  routine, output number formatting, column names or order. Moves and signatures only.
* No wholesale renaming of physics functions; `(conductor, …)` stays (only
  `simulation → ctx`).
* No regeneration of `tests/reference_data`; no edits to `models/` decks or results; driver
  scripts are touched only by the final import sed.
* Excel input: no new features, no removal; it is one frozen adapter.
* No dependency-injection framework, no abstract base classes everywhere:
  `typing.Protocol` plus constructor injection at one composition root.
* Do not flatten the nested topology dicts in one go; wrap, then migrate.
* Do not fix latent bugs in dead branches without a test proving they are dead or that the
  fix is byte-neutral.
* Do not make `transient_input` immutable: drivers write to it; it is the sanctioned hook.
* No stage combined with physics feature work.

## 5. Verification (every stage)

1. `pytest -m "not slow"` during work; full `pytest tests -v` (CASE_1/2/3 at 1e-5,
   checkpoint-restart SHA, `tests/verification`, about 4 minutes) before merge. CI already
   runs the full suite ([.github/workflows/tests.yml](../.github/workflows/tests.yml)).
2. `tests/tools/compare_trees.py`: the CASE_1 Output tree must be **bitwise** identical to
   the `main` baseline. Exceptions only where a PR documents why; none are planned — the
   Stage 3 dead-branch fixes, the Stage 4c path resolution and the Stage 8 mutation fix must
   still be byte-identical.
3. `lint-imports` passes; the `tests/test_architecture.py` baselines are strictly shorter
   than before the stage.
4. After Stages 4, 5 and 6: smoke-run `models/HELIAS-VIPER/1d_model/run_1d_model.py` and
   one network-coupled 2D model with a reduced `TEND` (they exercise every extension point),
   plus the GUI load path.
5. `ruff` and `mypy` on moved modules.

## 6. Relative size, value and risk

| Stage | Size | Value | Risk |
|---|---|---|---|
| 0 ratchet, markers, tree compare | S | enables everything | none |
| 1 import side effects, dumps, GUI fix | S–M | quick wins | low |
| 2 `utility_functions` split + import-linter | XL (moves) | removes the hub cycle | low |
| 3 fluid port + Environment | M | CoolProp and raw dict out of the domain | medium |
| 4a / 4b / 4c input extraction | M / L / M–L | testable domain objects | medium–high (4c) |
| 5 observers / output | L | Conductor stops doing I/O | medium (SHA guard) |
| 6 application layer + facade | L | real use-case layer | medium |
| 7a / 7b / 7c state objects | M each | de-god-object | low |
| 8 material protocol | M–L | removes string dispatch | medium |
| 9 drivers + tighten | M | finish line | low |

## 7. Critical files for implementers

* [source_code/simulation.py](../source_code/simulation.py) — orchestrator to split into
  `application/` + facade; hook points at lines 389-648.
* [source_code/conductor/conductor.py](../source_code/conductor/conductor.py) — file-based
  constructor (:114), I/O in `initialization` (:845-864) and `post_processing` (:1173),
  `simulation.*` reads.
* [source_code/utility_functions/transient_solution_functions.py](../source_code/utility_functions/transient_solution_functions.py)
  — numerics core to move; dumps :820/:880; prints :165/:286.
* [source_code/components/solid/solid_component.py](../source_code/components/solid/solid_component.py)
  — `simulation` parameter :205-214; lazy reads :398/:570/:697; CSV :116-140;
  `deal_with_flag_IOP_MODE` :1167-1198.
* [source_code/components/fluid/fluid_component.py](../source_code/components/fluid/fluid_component.py)
  — per-assembly workbook reads :274/:398/:506.
* [source_code/environment/environment.py](../source_code/environment/environment.py),
  [source_code/interfaces/coolprop_interface.py](../source_code/interfaces/coolprop_interface.py).
* [source_code/conductor/input_loader.py](../source_code/conductor/input_loader.py),
  `input_validator.py`, `components/**/*_inputs.py`,
  [source_code/interfaces/yaml_input_registry.py](../source_code/interfaces/yaml_input_registry.py).
* [pyproject.toml](../pyproject.toml) — import-linter contracts, pytest markers, ruff `T20`.

---

## Appendix — Deferred: a real root package

Optional later stage, to be decided after Stage 9: move `source_code/*` under
`opensc2/{core, model, application, adapters, drivers}` with the standard `package-dir`
layout and `python -m opensc2`; ship one release with a compatibility-shim package so
`models/**/run_*.py` keep working; then update the drivers. Benefits: layer names become
folder names and the stdlib name-clash constraint disappears. Cost: every import path in
about 25 driver scripts and in the tests changes once.
