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
