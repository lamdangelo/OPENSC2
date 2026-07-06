# Proposal: YAML-Based Input Format for OPENSC2

**Status:** IMPLEMENTED (stage 1, schema v1) — see "Implementation notes" at the end
**Date:** July 2026
**Scope:** replace the Excel workbook input files with a YAML-based format; keep
large numeric time series in CSV; provide a converter and a transition period with
dual support.

---

## 1. Motivation

The current input format consists of eight Excel workbooks per simulation
(`transitory_input`, `environment_input`, `conductor_definition`,
`conductor_<n>_input`, `conductor_<n>_operation`, `conductor_<n>_coupling`,
`conductor_grid`, `conductor_diagnostic`, plus `external_*.xlsx` time series).
Every one of the following pain points was hit in practice during the recent
refactoring and W7-X benchmark campaign:

1. **Scripted edits are hazardous.** Saving a workbook with `openpyxl` destroys
   cached formula values; the code then reads `None` and crashes. Every scripted
   edit needs a `soffice --headless --convert-to xlsx` repair round-trip — which
   itself silently does nothing when the output directory equals the source
   directory. Parameter scans and convergence studies should not require
   LibreOffice as a runtime dependency.
2. **Formulas are load-bearing but fragile.** `conductor_definition.xlsx` contains
   external-workbook-style references (`=[1]CONDUCTOR_files!A$1`) whose cached
   values the code depends on; a recalculation in the wrong context breaks the
   file invisibly.
3. **No meaningful version control.** Workbooks are binary blobs: `git diff` shows
   nothing, review is impossible, and merge conflicts are unresolvable. The W7-X
   tuning history (mesh, friction model, HTC flag, coupling switches) is invisible
   in git.
4. **Cell positions are part of the parser contract.** Readers rely on
   `skiprows=1`, header rows, and specific value columns (e.g. column E). Adding
   one new scalar (`TOLTIME`) requires knowing exact cell coordinates.
5. **Magic integers.** `IFRICTION: 206`, `Flag_htc_steady_corr: 2`, `IQFUN: -1`
   are meaningless without the manual, even though the refactored code now has
   explicit enumerations for all of them (`MethodFlag`, `HeatTransferModelType`,
   `HeatExcitation`, `ContactPerimeterFlag`, ...).
6. **Coupling matrices are error-prone.** Component-pair properties are entered as
   upper-triangular component×component matrices per sheet (11 sheets). An
   off-by-one landing on the diagonal (self-contact) cost a debugging session; the
   matrices are 95 % empty for typical conductors.
7. **Late, obscure failures.** There is no schema validation; a wrong entry
   surfaces as a `KeyError` or a physics anomaly deep inside the run.

## 2. Design Principles

- **YAML for configuration, CSV for data.** Scalars, flags, lists, and structure
  go to YAML. Large numeric tables (current waveform, external flow, spatial
  profiles) stay in CSV/TSV files referenced by path — YAML is a poor container
  for thousands of numbers, and CSV diffs cleanly.
- **Names, not magic numbers.** Every flag is written as the enumeration member
  name already defined in the code (`method: BDF2`,
  `friction_model: DARCY_FORCHHEIMER_POROUS_MEDIUM`). Integers remain accepted
  during the transition.
- **Sparse, symmetric couplings.** Component-pair properties are a list of pair
  records, not triangular matrices. Symmetry is by construction; the diagonal
  cannot be hit by accident.
- **Descriptive keys.** Key names follow the code base's full-word naming
  convention (`cross_section`, not `CROSSECTION`); a single alias table in the
  loader maps them onto the legacy raw-dictionary keys so the existing
  `build_*` dataclass constructors remain untouched.
- **Omission means default.** Optional settings (e.g. `tolerance` for the adaptive
  time stepping) are simply omitted; defaults live in one place in the loader.
- **One entry point.** `simulation.yaml` is the single file a user opens first; it
  references the per-conductor files.

## 3. Proposed File Layout

```
W7-X/
├── simulation.yaml              # transient settings + environment + conductor list
├── conductor_W7X.yaml           # one file per conductor: components, operations,
│                                #   couplings, grid, diagnostics
├── data/
│   ├── transport_current.csv    # replaces external_current.xlsx
│   └── ...                      # other external time series / profiles
```

For multi-conductor magnets each conductor gets its own file; shared component
definitions can be reused with YAML anchors (`&strand_type_A` / `*strand_type_A`)
or by factoring common blocks into an included file.

## 4. Example (W7-X, abridged but valid)

See `example_w7x_simulation.yaml` and `example_w7x_conductor.yaml` next to this
document for the full files; the essential shape:

```yaml
# simulation.yaml
simulation:
  name: W7X
  end_time: 25.0                # s
  time_stepping:
    method: BDF2                # BE | CN | GAL | BDF2 | AM4
    adaptivity: ERROR_CONTROLLED   # FIXED | LEGACY_EIGENVALUE | ERROR_CONTROLLED
    tolerance: 1.0e-3           # relative local-truncation-error target
    minimum_step: 1.0e-3        # s
    maximum_step: 0.1           # s

environment:
  temperature: 300.0            # K
  pressure: 1.0e5               # Pa

conductors:
  - file: conductor_W7X.yaml
```

```yaml
# conductor_W7X.yaml (excerpt)
conductor:
  name: W7X_conductor
  length: 146.0                 # m
  electric:
    method: BE
    time_step: 0.1              # s
    current:
      mode: FROM_FILE
      file: data/transport_current.csv
      initial: 15320.0          # A

components:
  - name: CHAN_1
    kind: FLUID_CHANNEL
    fluid: HELIUM
    cross_section: 3.7458e-5    # m^2
    hydraulic_diameter: 3.7641e-4  # m
    void_fraction: 0.375
    friction:
      model: DARCY_FORCHHEIMER_POROUS_MEDIUM
      multiplier: 1.0
    heat_transfer:
      model: DITTUS_BOELTER_PURE
    operation:
      inlet_temperature: 3.9    # K
      outlet_temperature: 3.93999962
      inlet_pressure: 1.72e6    # Pa
      # hydraulic BC type, flow direction, ... 

  - name: STR_MIX_1
    kind: MIXED_STRAND
    cross_section: 6.2004e-5
    superconductor:
      material: NBTI_W7X
      strand_count: 243
      strand_diameter: 5.7e-4
      critical_surface: {c0: 16.8512e10, Tc0m: 9.03, Bc20m: 14.61}
    stabilizer: {material: COPPER, RRR: 160.0}
    operation:
      magnetic_field:
        mode: LINEAR_WITH_TRANSIENT
        inlet: 5.64             # T
        outlet: 5.64
      heating:
        mode: SQUARE_WAVE_IN_TIME_AND_SPACE
        power: 40.0             # W  (2 J over 0.05 s)
        window_time: [0.05, 0.1]    # s
        window_position: [80.0, 80.1]  # m

  - name: Z_JACKET_1
    kind: JACKET
    material: AL6063
    cross_section: 3.58e-4

couplings:
  - between: [CHAN_1, STR_MIX_1]
    contact_perimeter: 0.3626   # m
    heat_transfer_coefficient: FROM_CORRELATION
  - between: [STR_MIX_1, Z_JACKET_1]
    contact_perimeter: 3.542e-3
    thermal_contact_resistance: 2.0e-3   # m^2 K / W
    electric_conductance_mode: IDEAL_PARALLEL   # THEA-style current sharing
  - between: [Z_JACKET_1, Z_JACKET_2]
    contact_perimeter: 0.064
    thermal_contact_resistance: 2.113e-3

grid:
  number_of_elements: 17000
  refinement:
    zone: [60.0, 100.0]         # m
    number_of_elements: 14450
    minimum_size: 1.5e-3        # m
    maximum_size: 0.1

diagnostics:
  spatial_distribution_times: [0.1, 0.15, 0.5, 1, 2, 4, 9, 10, 20]  # s
  time_evolution_positions: [80.2]                                   # m
```

Points worth noting in the example:

- The **coupling list** replaces eleven matrix sheets. Every record names its pair
  once; unspecified pairs are uncoupled. The THEA-style jacket current sharing —
  one obscure cell in the old format — becomes the self-explanatory
  `electric_conductance_mode: IDEAL_PARALLEL`.
- The **heating block** groups what the workbooks scatter over five flags
  (`IQFUN`, `Q0`, `TQBEG`, `TQEND`, `XQBEG`, `XQEND`).
- Scientific notation, comments, and trailing lists all diff cleanly in git.

## 5. Validation

Two layers, introduced in this order:

1. **Structural validation at load time (no new dependency).** The YAML loader
   emits the same raw dictionaries the Excel readers produce today, so the
   existing dataclass constructors (`build_conductor_inputs`, component
   `*_inputs.py` loaders) and the existing `input_validator` checks keep working
   unchanged. Unknown keys raise immediately with the file name and key path —
   already a large improvement over silent cell misreads.
2. **Schema validation (optional second step).** A JSON-Schema document generated
   from the existing dataclasses (or a `pydantic` model layer, if the dependency
   is acceptable) gives editor autocompletion (`yaml-language-server` schema
   pragma), range checks, and precise error messages before the run starts.
   This also yields free documentation: the schema *is* the input reference.

## 6. Integration Architecture

The refactored loader is already format-agnostic in the right place: every reader
returns a plain `dict` that feeds a dataclass constructor. The proposal adds one
seam:

```python
class InputSource(Protocol):
    def load_transient_settings(self) -> dict: ...
    def load_conductor_definition(self, index: int) -> dict: ...
    def load_component_inputs(self, ...) -> dict: ...
    # ... one method per current read
```

with two implementations: `ExcelInputSource` (the current code, unchanged) and
`YamlInputSource` (new, ~300 lines including the alias table
`yaml_key -> legacy raw key`). `Simulation`/`ConductorInputLoader` pick the source
by discovery: if `simulation.yaml` exists in the input directory it wins,
otherwise the Excel path is used. No physics code changes at all.

## 7. Migration Plan

1. **Converter first.** `tools/convert_inputs_to_yaml.py` reads an existing input
   directory (read-only — no `openpyxl` save, no repair round-trip needed) and
   writes the YAML equivalent, including the coupling-matrix → pair-list
   transformation and `external_*.xlsx` → CSV extraction. Run it on the TDD
   cases and W7-X; commit the YAML files next to the workbooks.
2. **Dual support** for at least one release: both formats load through the
   `InputSource` seam; the regression suite runs each TDD case in both formats
   and asserts identical solver state after 30 steps (the bit-compatibility gate
   already used throughout the refactoring).
3. **Flip the examples** to YAML as the documented primary format; keep the Excel
   reader available but frozen.
4. **Deprecate** the Excel path once users have migrated (announce, warn on load,
   remove in a later major version).

## 8. Open Questions

- **Units:** the proposal keeps the SI-only convention with units in comments.
  An explicit `{value: ..., unit: ...}` form (with `pint` conversion) is possible
  but adds complexity and a dependency; suggest deferring.
- **Inline vs. referenced conductors:** for single-conductor cases everything
  could live in one file; the two-file split keeps multi-conductor magnets
  manageable. Suggest supporting both (`conductors: [- file: ...]` or an inline
  mapping).
- **Where do user-defined Python hooks (custom current/field functions) get
  declared?** Suggest a `user_functions:` block naming an importable module path,
  replacing the current convention-based discovery.
## 9. Implementation Notes (stage 1, completed July 2026)

- **Delivered:** `source_code/interfaces/yaml_input_registry.py` (the
  `YamlInputRegistry` serving every configuration read; discovery rule: a
  directory containing `simulation.yaml` is YAML-driven) and
  `tools/convert_inputs_to_yaml.py` (read-only converter). Registry branches
  were added at every configuration read seam: `Simulation.__init__` /
  `conductor_instance`, `Environment`, `ConductorInputLoader` (definition,
  grid, coupling), the diagnostics reads, all five component input loaders,
  the component factory (format-agnostic `create_components` core), and
  `save_input_files` (metadata copy of the YAML files).
- **Schema v1 leaf keys are the legacy workbook names** inside
  `inputs:`/`operations:`/`grid:` blocks (converter emits verbatim), so the
  existing dataclass constructors work unchanged; the structural level
  (simulation block, couplings pair list, diagnostics lists, adaptivity
  names) already uses the descriptive form. The full leaf-key rename via
  alias table is schema v2.
- **Regression:** twin W7-X directories (Excel-only vs Excel+YAML), 50 time
  steps on the 17k mesh: all 61 output files **bit-identical**. CASE_1
  converts and instantiates; CASE_2's workbooks are mid-update and fail
  identically in both formats (faithfully reproduced, not converted in
  place).
- **Number-as-text repair:** workbook cells holding numbers as text (e.g. a
  cross section entered as `'6.2004E-5'`) are converted to real YAML
  numbers; genuine strings (material names, `BE`) are untouched. The Excel
  path keeps tolerating such cells unchanged, for backward compatibility.
- **Empty-cell semantics:** an empty value cell becomes YAML `null` and is
  served back as NaN, exactly matching what the pandas readers deliver
  (some inputs are legitimately empty, e.g. an unused
  `ELECTRIC_TIME_STEP`).
- **pandas lesson:** the converter reads cells with openpyxl directly —
  `pd.read_excel` coerces mixed boolean/integer workbook columns
  inconsistently depending on `dtype` (a `TRUE` cell can become `1`, an
  integer `1` become `True`).
- **Caution — precedence:** when both formats are present, YAML wins. After
  editing a workbook in a converted directory, re-run the converter (or
  delete the YAML files), otherwise the stale YAML silently overrides.

## 10. Open Questions (original)

- **YAML dialect:** plain `yaml.safe_load` (PyYAML — one small new dependency,
  now installed in the project venv) is sufficient; no custom tags required.
  `!include` convenience can be added later without breaking files that avoid it.
  One dialect gotcha the loader must guard against: under PyYAML's YAML-1.1
  rules, scientific notation **without a decimal point** (`1e-3`) parses as a
  *string*, while `1.0e-3` parses as a float. The loader should coerce
  numeric-typed fields with a clear error message (or use `ruamel.yaml` in
  YAML-1.2 mode, where both forms are floats); the example files consistently
  use the decimal-point form.
