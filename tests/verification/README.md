# Verification suite: hydraulic network and field-circuit coupling

Five cases compare OPENSC2 simulation results against closed-form analytical
solutions. All runs use the production entry points — the lumped-network
cases drive `HydraulicNetwork.solve_steady_state()` / `step(dt)` directly,
the channel and coupled cases run `Simulation(run_dir).run()` on generated
YAML inputs with fixed time stepping, imposing time-dependent boundary values
through the documented per-step callback — and a **constant-property fluid**
(`FluidType.CONSTANT`, served by `ConstantFluidProperties` in
`interfaces/coolprop_interface.py`: constant rho0, kappa_T, mu, cp, beta = 0)
so that every model has exactly the constant coefficients the analytical
solutions assume. All quantitative checks use laminar (Hagen-Poiseuille)
friction, where the branch resistance R = 32 mu l / (D^2 rho A) is exactly
linear; the network's `LAMINAR_REYNOLDS_FLOOR` guarantees that exact value at
arbitrarily low flow, and every case asserts or sizes its Reynolds numbers
inside the laminar band.

Run with `.venv/bin/python -m pytest tests/verification -v` (~4 min). The
companion scripts `scripts/verification/run_v*.py` reproduce each case and
save simulation-vs-analytics figures with error subplots to
`scripts/verification/output/`; the tests do not depend on the scripts.

## V1 — Steady Kirchhoff check (Wheatstone bridge)

`test_v1_kirchhoff.py`. A supply and a return reservoir feed an unbalanced
Wheatstone bridge of five laminar pipes through two connection pipes (four
zero-volume junction nodes). `solve_steady_state()` is compared against the
reduced nodal system `A_int G A_int^T p = -A_int G A_res^T p_res`
(G = diag(1/R), incidence +1 leaving / -1 entering) assembled by hand and
solved with numpy. Asserted: node pressures and all seven branch flows to
**1e-8 relative** (the problem is exactly linear, so agreement is limited
only by float64 round-off ~1e-15; the tolerance simultaneously certifies the
exact Hagen-Poiseuille reduction of the solver's pipe law), and nodal mass
balance `|A_int mdot| < 1e-12 max|mdot|` (exact mass conservation of the
returned solution). Reference: Kirchhoff circuit laws (any circuit-theory
text).

## V2 — Hydraulic RLC: surge-tank mass oscillation

`test_v2_rlc.py`. Reservoir -> laminar pipe (inertance L = l/A, resistance R)
-> dead-ended node of volume V (capacitance C = V rho0 kappa_T), released
from a 1 kPa pressure offset at zero flow; sigma/omega_0 = 0.05. Reference:
the underdamped series-RLC free response `p - p_res = dp0 e^{-sigma t}
[cos(omega_d t) + (sigma/omega_d) sin(omega_d t)]`, sigma = R/(2L), omega_d =
sqrt(1/(LC) - sigma^2) (Wylie & Streeter, *Fluid Transients in Systems*,
Prentice Hall 1993, surge-tank oscillation with linearized friction).
Asserted: (a) BDF2 at omega_d dt = 0.01, relative L2 error over 8 periods
**< 1e-3** (the BDF2 phase drift (omega dt)^2 Phi/6 over Phi = 16 pi is
~8e-4, so the bound checks second-order accuracy with the right constant);
(b) fitted convergence orders over dt-halving within **0.2** of theory —
backward Euler 1, Crank-Nicolson 2, Galerkin (theta = 2/3 ≠ 1/2) 1, BDF2 2 —
with fixed time stepping guaranteed structurally (`step(dt)` never adapts the
caller-supplied step); (c) sigma and omega_d extracted by exp-cosine
curve_fit within **0.5%**; (d) qualitative only: a quadratic valve
(tangent-linearized by the solver) yields an oscillatory decay with strictly
decreasing envelope and period within 5% of 2 pi/omega_0.

## V3 — Field model alone: water hammer and eigenfrequencies

`test_v3_water_hammer.py`. The channel's (v, p, T) discretization linearizes
to the fluid transmission line with inertance per length l = 1/A and
capacitance per length c = A rho0 kappa_T, giving wave speed a =
1/sqrt(rho0 kappa_T) (beta = 0 makes kappa_s = kappa_T) and characteristic
impedance **Z = sqrt(l/c) = a/A** in (p, mdot) variables — the density
cancels (Joukowsky dp = rho a dv with mdot = rho A v; dimensionally a/A =
Pa/(kg/s)). Reference: Wylie & Streeter 1993, ch. 1-3 and 8. Asserted, at
401 elements with CFL ≈ 1.27: (a) after instantaneous outlet closure of a
steady laminar flow, the first closed-end pressure plateau equals
p0 + Z mdot0 within **1%** and lasts 2L/a within **2%** (half-amplitude
crossings; the scheme's upwind front smear ~sqrt(2Lh) = 0.7 m is symmetric
about the ideal fronts, and the low-viscosity acoustic fluid keeps the
line-packing tilt at 0.1% of the rise); (b) the first three quarter-wave
eigenfrequencies f_n = (2n-1) a/(4L), excited by a raised-cosine inlet pulse
into the closed line and extracted from a 30 s Hann-windowed FFT of the
midpoint pressure with parabolic peak interpolation, within **1%** each
(measured 0.005% / 0.07% / 0.21%; Crank-Nicolson keeps the record
time-dissipation-free); (c) mesh convergence over {51, 101, 201, 401}
elements: the relative L2 distance of the closed-end trace from the exact
square wave (amplitude Z mdot0, period 4L/a) decreases strictly under
refinement (0.293 / 0.247 / 0.209 / 0.182, the sqrt(h) front-smear scaling).
The FFT-peak error itself was *not* used for the convergence evidence: the
spatial dispersion error in f_1 (~1e-4 relative at 51 elements) lies far
below the ~0.5% peak-estimator resolution, so a peak-error trend would be
estimator noise.

## V4 — Coupling: transmission line with lumped termination

`test_v4_transmission_line.py`. The production coupled solve (YAML
`hydraulic_network:` section -> `build_coupled_network` -> per-step bordered
Schur solve) terminates the channel outlet port on a zero-volume junction
discharging through a single linear resistance R_term into a reference
reservoir. A raised-cosine pressure pulse (1 kPa, 2 m ≈ 80 elements)
launched from the inlet is recorded at the 3L/4 probe; the reflection
coefficient is measured as the signed ratio of time-integrated pulse areas
over disjoint incident/reflected windows computed from a and the geometry
(pulse area is invariant under the scheme's diffusive spreading, unlike the
peak amplitude, whose ratio is 0.89 even for a perfect reflector).
Reference: Gamma = (R_term - Z)/(R_term + Z), standard transmission-line
theory (Wylie & Streeter 1993, ch. 3). These four runs are the direct
quantitative test of pressure continuity and mass conservation at the port
and of the Schur-complement condensation — any defect there appears as a
spurious reflection. Asserted: matched R_term = Z, residual in the reflected
window **< 2%** of the incident peak (measured 0.3%); closed
R_term = 1e12 (the `RELIEF_VALVE_BLOCKED_RESISTANCE` scale), Gamma = +1 ±
**0.02** (measured +0.998); reservoir R_term = 1e-4 Z, Gamma = -1 ± **0.02**
(measured -0.997); R_term = 3Z, Gamma = 0.5 ± **0.02** (measured +0.499).

## V5 — Segregated thermal update: first-order lag

`test_v5_thermal_lag.py`. A single node of volume V receives a constant
external inflow mdot_in at T_in (production `external_mass_sources` +
`external_source_temperatures` interface) and discharges through a linear
valve; the hydraulic state starts at its exact fixed point. Reference:
T(t) = T_in + (T0 - T_in) e^{-lambda t}, lambda = mdot_in/(rho0 V) — the
segregated node-energy update reduces exactly to backward Euler on this
scalar equation (cp cancels). Asserted: relative L2 error **< 1e-3** at
lambda dt = 2.5e-3 over 5 time constants; fitted order **1 within 0.2**
under dt-halving anchored at lambda dt = 0.02; and, because beta = 0 makes
the V rho beta dT/dt expansion source vanish identically, the node pressure
stays constant to **1e-10 relative** and the discharge flow to 1e-12 kg/s
absolute throughout the transient (no spurious thermal-hydraulic feedback).

## Deviations from the original specification (agreed during planning)

- **V5 accuracy step.** The spec asked for L2 < 1e-3 at lambda dt = 0.01,
  but backward Euler's error constant makes that analytically unreachable
  (~3.6e-3). The accuracy assertion runs at lambda dt = 2.5e-3 (expected
  error ~9e-4) with the tolerance kept at 1e-3; lambda dt = 0.01 anchors the
  order study instead.
- **V3 mesh-convergence metric.** The spec asked for a monotone decrease of
  the first FFT-peak error over the meshes; that error sits below the
  estimator resolution (see V3 above), so the time-domain L2 distance from
  the exact square wave is used as the convergence evidence, while the
  3-peak/1% FFT assertion is kept at 401 elements as specified.

## Open discrepancies

None — all asserted tolerances hold. One code finding surfaced while
building the suite (a robustness limitation rather than an accuracy
discrepancy, so it is recorded here instead of as an xfail):

- **Exactly zero flow crashes the channel initialization/assembly.** The
  Blasius turbulent correlation `0.316 Re^-0.25`
  (`hydraulics/turbulent_friction.py`) has no zero-Reynolds guard (the
  laminar correlations do), so a channel initialized with exactly zero
  mass-flow rate produces inf in `gen_flow` and NaN in the first assembled
  system. V3(b) and V4 therefore seed the quiescent line with 1e-12 kg/s
  (a 4e-7 Pa bias, seven orders below the pulse amplitudes). A zero-Re guard
  in the turbulent correlations would remove the need for the seed; it was
  not added here because the suite's ground rule forbids touching solver
  code paths.

## Formulation suite (velocity vs mass-flow under network coupling)

`tests/verification/formulation_suite/` compares the velocity (v, p, T) and
the explicit mass-flow (mdot, p, T) formulations on a W7-X NbTi CICC channel
(`W7-X/comparison_1D/conductor_CONDUCTOR_100.yaml`), three conductors of
20/30/40 m in parallel between two 0.5 L volumes fed through linear valves
from fixed-pressure reservoirs (5.0 / 4.5 bar at the volumes, 4.5 K).
Independent references live in `tests/reference/` (steady compressible
network ODE, linear transmission line with characteristic boundary closure,
manufactured-solution residual), each with solver-free unit tests
(`pytest tests/reference`).

Tiers: 1 steady flow split, 2 low-Mach acoustic step, 3a symmetric heating
(midpoint flow must vanish), 3b asymmetric heating with Richardson
extrapolation, 4 manufactured solution (through the opt-in
`Conductor.fluid_source_callback` hook). Mass-balance diagnostics (port
residual with the mass flow exactly as passed to the network, interior
inventory defect) are recorded every step.

    .venv/bin/python tests/verification/formulation_suite/driver.py --tier all --quick   # coarsest level
    scripts/run_formulation_matrix.sh                                                     # full matrix (hours)
    .venv/bin/python tests/verification/formulation_suite/driver.py --no-run --metrics --summary
    .venv/bin/python scripts/make_plots.py                                                 # figures/ from results/

Only the single-pass mode exists (the solver has no Picard sub-iteration).
See `results/summary.md` for the numbers and the documented deviations.
