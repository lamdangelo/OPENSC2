# Developer notes: the (ṁ, p, T) hydraulic formulation

*(and the autosave/restart infrastructure it ships with)*

## What and why

The 1D channel hydraulics historically solves the quasilinear compressible
system in the primitive variables **W = (v, p, T)** per channel. The hydraulic
field-circuit coupling, however, works in node pressures and **branch mass
flows**, and had to reconstruct the port flow as ṁ = ρ_old·A·v (frozen-density
Picard linearization). The selectable **W′ = (ṁ, p, T)** formulation makes the
coupling variable a native unknown: the port coupling coefficient becomes the
exact ±1, imposed-mass-flow boundary conditions become exact Dirichlet rows
(instead of v = ṁ/(ρ_lagged·A)), and the network exchange loses its
frozen-density defect at the interface.

Selection: `conductor.inputs.hydraulic_formulation` = `auto` (default) |
`velocity` | `mass_flow`. `auto` resolves in
`Simulation.conductor_initialization`, *before* any conductor initialization
work (the solution seeding depends on it): a conductor with a declared
`hydraulic_network` port resolves to `mass_flow`, otherwise `velocity`.
Explicit values win; explicit `velocity` on a coupled conductor warns.
All conductors sharing one network must resolve identically
(checked in `build_coupled_network`).

## The transformation (hydraulics/formulation.py)

With ṁ = ρ(p,T)·A·v and A constant per channel, the change-of-variables
Jacobian M = ∂W/∂W′ and its analytic inverse are

```
      | 1/(ρA)   −v·κ_T    v·β |            | ρA   ρA·v·κ_T   −ρA·v·β |
  M = |   0         1       0  | ,   M⁻¹ =  |  0       1          0   |
      |   0         0       1  |            |  0       0          1   |
```

with κ_T = (1/ρ)(∂ρ/∂p)_T the isothermal compressibility and
β = −(1/ρ)(∂ρ/∂T)_p the expansion coefficient (both already evaluated per
node and per Gauss point every step by the CoolProp property pass — the same
EOS backend as everything else, no second property path).

**Route: conjugation, not re-derivation.** The solver assembles the
per-Gauss-point flux-Jacobian (AMAT), upwind/diffusion (KMAT) and
source-Jacobian (SMAT) blocks exactly as before; when the resolved
formulation is `mass_flow`, one hook in `assemble_thermal_hydraulic_system`
applies the similarity transform X → M⁻¹XM in place, with coefficients
frozen at the same state the blocks were built from. The two formulations
therefore cannot drift apart, and the eigenvalues {v, v±c} are invariant
(unit-tested to 1e-9, kernel vs dense reference to 1e-12).

**Known, measured limitation of the frozen-M route.** The exact change of
variables W = M(W)·W′ produces coefficient-derivative terms
M⁻¹(∂M/∂t + A·∂M/∂x)·W′ that the frozen similarity omits (M is
state-dependent, so the transform is only exactly equivalence-preserving
for constant M). These terms vanish in (near-)steady states — the coupled
degenerate limit agrees with the velocity formulation to 1e-6 and the
steady axial mass-flow uniformity to below 5e-3 — but during density
transients they are O(v·Δρ/ρ). Measured on the 100 W/m CASE_1 test pulse:
pressure agreement 6e-4, temperature 1–2e-2, flow fields 1.2–1.4e-1,
*invariant under both mesh refinement (50→100 elements) and time-step
refinement (0.1→0.025 s)* — the fingerprint of an omitted continuous term
rather than a discretization error. If transient cross-formulation
equivalence is required, the conjugation would have to be extended with
the ∂M source terms (lagged ∂M/∂t, Gauss-point ∂M/∂x) — a deliberate
follow-up, not part of this change.

**Robustness on violent transients.** Both formulations have blow-up
thresholds on the boosted CASE_1 pulse (INTIAL = 1, 50 elements,
Δt = 0.1 s): the velocity path loses the inlet-temperature boundary
condition under the backflow sign flip at ~500 W/m (invalid negative
temperature), and the mass-flow path on the coupled bypass loop reaches an
invalid negative pressure at the same 500 W/m pulse (t = 0.7 s) where the
velocity path survives; at 300 W/m both complete. For production quench
transients on network-coupled decks (which `auto` resolves to `mass_flow`)
this means: validate the replay, and pin `hydraulic_formulation: velocity`
in the deck if the mass-flow run proves fragile on the pulse of interest.

Why the rest of the system is untouched:

* the fluid transient block is the identity, and M⁻¹·I·M = I;
* the fluid rows of the load vector are identically zero (all heat sources
  live on solid rows), so b′ = M⁻¹b = b;
* boundary-condition rows are overwritten *after* assembly and are never
  conjugated — in `mass_flow` mode they carry the native Dirichlet value;
* one assembly per step feeds both sides of the θ/BDF2 schemes and the
  history vector is natively W′, so no cross-step transformation-consistency
  question arises (the standard frozen-coefficient treatment).

Because M and M⁻¹ differ from the identity only in the flow-slot row/column
of each channel triple, the global transform is one row operation plus one
column operation per channel; the row phase must complete for **all**
channels before any column phase starts (fluid–fluid interface blocks couple
two channels' triples). Solid rows carry no fluid-velocity columns
(fluid–solid coupling is temperature-only), so they are bit-invariant.

## Semantics of the flow slot

| concern | velocity mode | mass_flow mode |
|---|---|---|
| slot content | v | ṁ |
| write-back | `node_fields.velocity` native; ṁ = ρAv derived post-step | `node_fields.mass_flow_rate` native; v = ṁ/(ρ_new·A) in the per-step refresh |
| Reynolds | ρ v D_h/μ | density-free ṁ D_h/(A μ), wrapped in abs |
| ṁ boundary condition | v_b = ṁ/(ρ_lagged·A) | exact Dirichlet `known = ṁ` |
| port coupling C-block | −σ·ρ_old·A on v column | −σ on ṁ column (exact) |
| LTE controller floor (slot) | 1 m/s | ρ_max·A·(1 m/s) — the dimensional image, preserving controller aggressiveness across channel sizes |

Friction factors and HTC correlations consume only Re (and derived fields),
never the raw slot — both formulations feed them the same physical
quantities. The velocity field remains first-class everywhere (outputs,
plots, sign tests): it is refreshed every step from the native ṁ.

Both v and ṁ appear in every output regardless of formulation
(spatial distributions, per-field time evolutions, inlet/outlet records).

## Autosave / restart (utility_functions/checkpoint.py)

`simulation.autosave_interval: N` stores a checkpoint of the complete
evolving state every N accepted steps (`Checkpoints/checkpoint_step_*.npz`,
atomic write, two most recent kept). The time-evolution buffers are flushed
first, so output files are always complete up to a checkpoint.
`simulation.restart: true` re-runs the deterministic initialization, then
overwrites the evolving state from the newest checkpoint and truncates the
output files to the checkpoint time; the continuation is **bit-identical**
to an uninterrupted run (regression-tested for plain and network-coupled
runs). Restart targets *interrupted* runs: a finalized run (post-processing
consumed the raw spatial saves) is refused with a clear message.

**Formulation policy on restart:** the checkpoint records the resolved
formulation per conductor; restarting under a different formulation FAILS
with a message naming both values. State is deliberately *not* converted
through M/M⁻¹: the stored history levels (BDF2/θ, LTE predictor) belong to
the discretization that produced them, and a converted state would be
silently inconsistent with them.

## When to prefer which formulation

`mass_flow` is the natural choice for network-coupled runs (native coupling
variable, exact port block, exact ṁ boundary conditions) — and is what
`auto` picks there. `velocity` remains the default for standalone conductor
runs and for legacy comparison work (THEA replication, historical decks),
where bit-identical reproduction of earlier results matters. In steady and
near-steady conditions the two agree to solver roundoff / discretization
order; during fast density transients they differ at O(v·Δρ/ρ) in the flow
fields (see the limitation above), so quantitative transient comparisons
against earlier velocity-formulation results should stay on the velocity
formulation, and within one study the formulation should be kept fixed.

## Explicit (ṁ, p, T) assembly (`explicit_mass_flow_formulation: true`)

*(added 2026-09-08; code in `hydraulics/mass_flow_equations.py`, tests in
`tests/test_mass_flow_explicit*.py`)*

A second route to the same (ṁ, p, T) unknowns: the Gauss-point blocks are
assembled **directly from the written-out coefficients** instead of
conjugating the velocity-form blocks at runtime. Selected by the boolean
conductor input `explicit_mass_flow_formulation` (YAML; legacy key
`EXPLICIT_MASS_FLOW_FORMULATION`; default `false`). Resolution
(`hydraulics/formulation.py::resolve_hydraulic_formulation`):

| `hydraulic_formulation` | `explicit_mass_flow_formulation` | resolved |
|---|---|---|
| `auto` / `velocity` (uncoupled) | `false` | `velocity` (unchanged) |
| `auto` (coupled) / `mass_flow` | `false` | `mass_flow` — similarity transform (unchanged) |
| `auto` / `mass_flow` | `true` | `mass_flow_explicit` — explicit assembly |
| `velocity` | `true` | error (contradictory deck) |

With the flag `false` every existing deck is bit-identical (checked on the
full CASE_1 velocity run and on 2 s velocity / transform runs: 70 output
files each, zero differing hashes). The internal enum value
`HydraulicFormulation.MASS_FLOW_EXPLICIT` is not an input value; the
checkpoint compatibility check works on it unchanged (a restart across
routes is refused like any other formulation change). All conductors of one
hydraulic network must resolve to the same route.

### Equations implemented

With v = ṁ/(ρA) a *derived* quantity (evaluated, like every coefficient, at
the previous time level: the nodal velocity is rebuilt as ṁ/(ρA) after every
step and carried to the Gauss points by the same two-node average as p and
T), the quasi-linear system ∂ₜW + K̃ ∂ₓW + S̃ W = q̃ per channel is

```
∂ₜṁ + (2ṁ/(ρA)) ∂ₓṁ + (A − κ_T ṁ²/(ρA)) ∂ₓp + (β ṁ²/(ρA)) ∂ₓT + (2f|ṁ|/(ρA D_h)) ṁ = 0
∂ₜp + (c²/A) ∂ₓṁ − ((γ−1) ṁ/(ρA)) ∂ₓp + (β c² ṁ/A) ∂ₓT − 2fφ|ṁ|ṁ²/(ρ²A³D_h)     = q̇_p
∂ₜT + (φT/(ρA)) ∂ₓṁ − (φTκ_T ṁ/(ρA)) ∂ₓp + (γ ṁ/(ρA)) ∂ₓT − 2f|ṁ|ṁ²/(ρ³A³c_v D_h) = q̇_T
```

```
     | 2v            A(1 − ρκ_T v²)   ρAβv²   |          | 1              0 0 |
K̃ = | c²/A          −(γ−1)v          ρβc²v   |   S̃ = (2f|v|/D_h) | −φv/A          0 0 |
     | φT/(ρA)       −φTκ_T v         γv      |          | −v/(ρAc_v)     0 0 |
```

Time discretisation, element integration (consistent Galerkin mass matrix),
θ/BDF2 weighting, boundary conditions, network ports and the friction
treatment (|ṁ| frozen, ṁ implicit, kept in S̃) are the existing ones: the
explicit route only replaces the Gauss-point builders of the fluid rows
(`build_amat_mass_flow`, `build_kmat_fluid_mass_flow`,
`build_smat_fluid_momentum_mass_flow`, `build_smat_fluid_energy_mass_flow`,
`build_smat_fluid_interface_momentum_mass_flow`) inside
`assemble_thermal_hydraulic_system`. The temperature-row exchange builders
(`thermal/energy_equation.py`: fluid–fluid energy, fluid–solid) are
formulation-invariant on their p/T columns and are reused unchanged.

**Direct vs derived properties.** Direct from the property library at the
Gauss state: ρ, β, κ_T, c_v, c_p, c. Derived: γ = c_p/c_v; φ = β/(ρκ_T c_v)
(Grüneisen, the same derivation as the velocity path). The identities
ρc²κ_T = γ (Reech) and 1 + φβT = γ (generalised Mayer) are used to write
the diagonal advection entries as 2v, −(γ−1)v, γv; they hold to 4e-16 with
the direct HEOS flash and to ~1e-8 with the production bicubic property
tables on 4–40 K / 3–10 bar helium samples (measured; the table accuracy
degrades towards the pseudo-critical peak). No ideal-gas relation is used.

**Heat-source consistency.** The fluid rows of the load vector are
identically zero; q̇_p and q̇_T exist only as entries of the source-Jacobian
S multiplying (p_j, T_j, p_k, T_k, T_solid). They are populated by
*different routines* — pressure row of the channel–channel exchange in
`hydraulics/momentum_equation.py`, temperature row in
`thermal/energy_equation.py`, both rows of the channel–solid exchange in
`thermal/energy_equation.py` — but from the same Gauss fields and the same
φ, so q̇_p = φρc_v q̇_T holds column by column and κ_T q̇_p − β q̇_T = 0 is a
statement about S columns. The explicit route enforces it structurally (the
ṁ row carries no temperature column); the only exchange term of the ṁ row
is the momentum ±K₂ = ±K₁λ_v v carried by the exchanged mass at open
interfaces (closed form of the M⁻¹-combined rows: the K₃ − vK₂ energy terms
cancel through κ_Tφρ = β/c_v, the recovery terms c²/φ and φc_vT combine to
ρc²κ_T − βφT = 1). Debug-mode assertion: `OPENSC2_DEBUG_CHECKS=1` runs
`check_source_consistency` on the assembled block every step (temperature
columns to 1e-12, partner-pressure residual −K₁/(ρA) and the ṁ-row ±K₂ to
the identity level). Measured residuals on real helium states: 2e-16 /
7e-16 / 0.

### Relation to the transform route

Per Gauss point the explicit K̃ and S̃ are entry-for-entry M⁻¹AM and M⁻¹SP;
on the first assembled CASE_1 step the flux and source blocks of the two
routes agree to 5.6e-16 / 1.9e-15 (row-scaled) with the direct HEOS flash
and to the property-identity level, 4.4e-8 / 2.4e-7, with the production
tables. **The two routes differ only in the stabilisation block**: the
explicit route keeps the diagonal artificial-diffusion structure of the
velocity builder (Δz·u/2, Δz(u+c)/2 on the ṁ row), whereas the transform
conjugates it, K′ = M⁻¹KM, which adds the ṁ-row off-diagonals
(ṁ,p) = ρAvκ_T(k_p − k_ṁ) = −ρAvκ_T Δz c/2 and (ṁ,T) = ρAvβ Δz c/2 (verified
entry by entry). These are not small: weighted by the state scales they
amount to 11 % ((ṁ,p)·p vs k_ṁ·ṁ) and 39 % ((ṁ,T)·T) of the ṁ-row diagonal
term on the CASE_1 state. Consequences, measured on the 100 W/m CASE_1 pulse (2 s,
pressure-drop BCs) and reported by `test_mass_flow_explicit_vs_transform.py`:

| explicit vs transform, final state | 50 el | 100 el | 200 el | fitted orders |
|---|---|---|---|---|
| CHAN_1 ṁ | 8.0e-3 | 6.5e-3 | 4.9e-3 | 0.31, 0.39 |
| CHAN_1 p | 2.4e-4 | 1.4e-4 | 6.7e-5 | 0.80, 1.02 |
| CHAN_1 T | 6.3e-4 | 7.8e-4 | 8.1e-4 | refinement-stable |
| CHAN_2 ṁ | 5.5e-3 | 4.4e-3 | 3.2e-3 | 0.31, 0.45 |
| CHAN_2 p | 2.4e-4 | 1.4e-4 | 6.7e-5 | 0.80, 1.02 |
| CHAN_2 T | 2.3e-4 | 3.0e-4 | 3.1e-4 | refinement-stable |

Time-step refinement (0.1 → 0.05 → 0.025 s, 50 el) leaves the deviation
unchanged (pressure exactly 2.354e-4 at all three), i.e. it is a spatial-
operator difference. Pressure converges at first order (the O(Δz) upwind
block), ṁ sub-linearly, temperature sits on a refinement-stable floor like
the velocity-vs-mass-flow floor documented above. On the steady
pressure-drop run (heating off) the routes differ by 2.3e-4 / 1.5e-4 in ṁ,
1.2e-6 in p, 9e-6 in T (CHAN_1 / CHAN_2): the boundary rows of the two
stabilisation blocks fix slightly different discrete steady states; on the
constant-property single channel the same difference is 1.1e-7.

### Verification results (2026-09-08)

* Unit tests (20 random helium states, 4–40 K, 3–10 bar, |v|/c ∈ [1e-4,
  1e-1], direct HEOS): eigenvalues of K̃ = {v−c, v, v+c} and K̃ = J⁻¹KJ to
  1e-10, trace 3v, ṁ row of S̃W exactly (2f|ṁ|/(ρAD_h))ṁ and bit-identical
  under doubled heat transfer, interface momentum exactly ±K₂,
  κ_Tφ − β/(ρc_v) = 0 to 4 ulp.
* Steady isothermal single channel (constant fluid, laminar, Δp = 200 Pa):
  ṁ vs Darcy–Weisbach with the code's own f to 3.9e-6 (the compressible
  convective acceleration, κ_TΔp ~ 2e-4, is the residual); velocity /
  transform / explicit mean ṁ agree to 8e-9. Axial ṁ uniformity: explicit
  5e-14, transform 2e-7, **velocity 2.0e-4** (its steady state is uniform in
  v, so ρAv carries the density variation).
* Low-Mach pressure pulse (constant fluid, 400 el, Δt = 5e-4 s): front speed
  between probes at 2.5 m and 7.5 m = c + 0.006 % for both formulations.
* CASE_1 (100 s, 200 el) per-channel flows at TEND [kg/s] against the
  committed reference (velocity matches to 1e-6; the reference itself is
  known to drift 2.8e-4 on velocity_out):

  | channel | quantity | reference | velocity | mass_flow | explicit |
  |---|---|---|---|---|---|
  | CHAN_1 | ṁ inlet | 8.398160e-03 | 8.398160e-03 | 8.398160e-03 | 8.398160e-03 |
  | CHAN_1 | ṁ outlet | 8.397018e-03 | 8.397018e-03 | 8.397027e-03 | 8.397994e-03 |
  | CHAN_2 | ṁ inlet | 1.248184e-02 | 1.248184e-02 | 1.248184e-02 | 1.248184e-02 |
  | CHAN_2 | ṁ outlet | 1.248296e-02 | 1.248297e-02 | 1.248296e-02 | 1.248200e-02 |

* Mass-conservation diagnostic (100 W/m pulse, 2 s, 50 el; per-step log of
  ∫ρA dz and ṁ_in − ṁ_out, right-endpoint quadrature): accumulated defect
  relative to the integrated inlet throughput — velocity 1.93e-2, transform
  1.44e-2, explicit 1.41e-2 (measured, not fixed).

### Open items

* **Stabilisation consistency between the formulations is not
  established** (marked `TODO` in `build_kmat_fluid_mass_flow`): the
  explicit route applies the velocity-form diagonal structure unchanged to
  rows of different dimension, the transform route carries the conjugated
  off-diagonals; neither has been shown to be the consistent choice, the
  coefficients were deliberately not retuned, and the measured
  cross-route differences above are entirely this block.
* The q̇_p / q̇_T consistency is guaranteed by construction where both rows
  are built from the same φ; it rests on Reech + Mayer for the
  partner-pressure columns of open interfaces, i.e. on the property table
  accuracy (~1e-8 sampled, degrading towards the pseudo-critical peak).
* The Gauss velocity used by the explicit builders is the two-node average
  of ṁ/(ρA), not ṁ_G/(ρ_G A) (O(Δz²) apart), to stay consistent with the
  reused temperature-row builders.
* The ρ_old·A·v reconstruction still exists on the velocity path only
  (`coupling.py` velocity branches of `mass_flow_into_node` /
  `_coupling_blocks`; provisional velocity in `reorganize_th_solution`).
