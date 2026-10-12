# Changelog

All notable changes to OPENSC² are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-09

First tagged release of the refactored OPENSC² fork maintained by the
Stellarator Reactor Studies group at IPP Greifswald. It supersedes the
`develop` branch of the upstream MAHTEP project (merged as pull request #1 on
2026-07-06) and bundles the complete refactoring, modernization and
performance work since then.

### Breaking changes

- The source tree is now the installable `opensc2` package. The `source_code/`
  directory and its top-level modules (`conductor`, `simulation`, `hydraulics`,
  ...) are gone. Every import needs the `opensc2.` prefix, for example
  `from opensc2.simulation import Simulation`. No compatibility shims are
  provided.
- Entry points: `python -m opensc2.drivers.cli <model_directory>` or the
  installed `opensc2` command for headless runs, `python -m
  opensc2.simulation_starter` for the GUI. Driver scripts that inserted
  `source_code` into `sys.path` must be updated.
- Existing editable installs must be refreshed with `pip install -e .`.
- Python 3.11 or newer is required (previously 3.8).
- An unknown `superconducting_material` name now raises at construction. It
  previously left the critical properties silently unset.
- Conductor run-time state is exposed only through its state objects
  (`conductor.time_state`, `.electric`, `.balances`, `.save_schedule`). The
  legacy aliases such as `conductor.cond_time` no longer exist.

### Added

- YAML input format (schema v2) as a fully interchangeable alternative to the
  Excel workbooks, with `tools/convert_inputs_to_yaml.py` and bit-identical
  results between the two formats. YAML takes precedence when both are present.
- Hydraulic network coupling: conductors are embedded in a lumped-element
  coolant circuit (manifolds, valves, lines) solved together with the
  distributed channels in a single Schur-complement system, including node
  enthalpy balances and thermal expansion sources.
- Mass-flow formulation `(mdot, p, T)` of the channel equations next to the
  original velocity formulation, selectable per deck, plus an explicit variant.
- Adaptive time integration: Galerkin and second-order backward-differentiation
  time steppers with error-controlled step adaption.
- Autosave checkpoints and restart of finalized or interrupted runs.
- Opt-in AC-loss heat sources: inter-strand coupling loss with relaxation
  model, eddy currents in jackets and stabilizers, hysteresis loss for NbTi,
  Nb3Sn and HTS stacks.
- Thermal coupling between neighbouring conductors through insulation patches
  (`transverse_coupling_file`), read through the external-input port.
- HTS stack component (`StackComponent`) with a steady-state electric solver
  for current distribution.
- External mesh files for two-dimensional conductor layouts (`mesh_type -1`).
- Steady-state and analytical verification suite (`tests/verification`),
  a constant-property fluid for analytical checks, and regression tests for
  the three TDD examples running in GitHub Actions.
- `tests/tools/compare_trees.py` for byte-wise comparison of output trees
  between commits.
- Architecture documentation (`docs/architecture.md`) with recipes for adding
  outputs, input formats, materials and external quantities.

### Changed

- Clean Architecture layering: domain packages (physics, numerics, runtime
  state), ports, application use cases, adapters (input, output, plotting,
  checkpoint) and drivers (CLI, GUI). Dependencies point inward only, enforced
  by `tests/test_architecture.py` and import-linter contracts.
- Fortran-heritage names and string-keyed dictionaries replaced by typed data
  classes and enumerations throughout.
- Superconductor correlations are `Superconductor` material objects with a
  common critical-surface interface; material-string dispatch is gone.
- In-memory mesh and contact tables are numpy arrays instead of pandas frames.
  The analytical inductance path works again under pandas 3.
- Sparse banded system assembly, vectorized element construction, LAPACK
  solvers and tabulated fluid properties: the W7-X quench benchmark
  (17 000 elements, 25 s transient) runs in about 19 minutes instead of hours.
- Regression references of the TDD examples re-baselined to the current
  physics (hydraulic network and mass-flow work); the refactoring itself is
  byte-identical to the previous `main`.

### Fixed

- CASE_2 and CASE_3 of the TDD examples run again: workbook `"None"` cells
  read as NaN, dangling laminar friction-factor enum member, laminar branch
  of the HTS CICC friction correlations, jacket insulation material `nan`,
  scalar bisection bounds for RE123 current sharing, nitrogen as coolant,
  optional strain input for stabilizer strands.
- Fluid-property evaluation robust against CoolProp returning arrays for
  single-element inputs.
- Headless runs no longer force the Tk matplotlib backend.

[2.0.0]: https://github.com/lamdangelo/OPENSC2/releases/tag/v2.0.0
