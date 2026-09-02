# sinbad-oracle-fenicsx status

**Updated:** 2026-09-01 (W7 lane 6, SV0-C5 D2-D5)
**Protocol:** `sinbad-oracle-protocol/1` (Sinbad-owned), adapter identity
`sinbad-oracle-fenicsx@0.11.0.post0`, `normalization_version = 1`
**Live environment:** `dolfinx/dolfinx:stable` (digest and toolchain in `INSTALL.md`), driven
through `scripts/dolfinx-image.sh`; every number below is a real run of the CLI in that image,
recorded verbatim in `tests/fixtures/recorded/`.

## Landed

### Shared adapter contract (all capabilities)

- `registry.py`: one `CapabilitySpec` per Sinbad case (capability id, case, model, dimension,
  module, observable id -> normalization-1 definition). The adapter answers unknown-capability,
  unsupported-observable and unaddressable-refinement questions from the registry without
  importing dolfinx.
- `outcome.py`: `SolveOutcome` (observables, mesh record, field records, solver notes) and
  `UnsupportedCase` (module-level typed refusal) between the adapter and capability modules.
- `evidence.py`: `<result>.evidence/{manifest.json,mesh.npz,solution.npz}`, schema
  `sinbad-oracle-fenicsx-evidence/1`: request, identity, full toolchain (dolfinx version + git
  commit, basix, ufl, PETSc, petsc4py, mpi4py, numpy, python), sha256 of every extraction script
  and of the package, mesh/solution sha256, normalization definitions, decimal observables,
  solver notes. A failure to retain evidence is a crash (no result file), never a bare number.
- `NORMALIZATION.md` is generated from the registry (`normalization_doc.py`) and pinned by
  `tests/test_registry.py::test_normalization_doc_is_generated_from_the_registry`.
- `scripts/dolfinx-image.sh` (run anything in the official image as the caller's uid) and
  `scripts/record_live_runs.py` (re-record the fixtures through the real CLI).

### D1 `poisson` (`01-poisson`) -- re-based on the shared contract, numbers unchanged

| level | energy | l2_error | h1_seminorm_error |
|---|---|---|---|
| [4,4] | 2.1158194762635505 | 0.07907545425172648 | 0.8385483442178129 |
| [8,8] | 2.374176221838155 | 0.021132773458254567 | 0.4317982830064737 |
| [16,16] | 2.4437400714546524 | 0.005377435009947058 | 0.2175363363595291 |

Identical to the values GX-CONTRACTS C11.18 recorded for the sealed `IndependentlyVerified`
claim (`test_recorded_fixtures.py::test_recorded_values_match_sinbads_own_contract_records`).

### D2 `nonlinear_heat` (`03-nonlinear-heat`) -- ran live, satisfied

BDF2 (step 0.05, final time 0.4, BDF1 start-up), SNES Newton per step (2-3 iterations),
plus a steady Newton companion; Q derived by UFL from `T_exact`, not copied from the case.

| level | total_energy | l2_error | h1_seminorm_error | nodal_l2_error | steady_l2_error | forcing_defect |
|---|---|---|---|---|---|---|
| [2,2] | 300.2114187500891 | 0.24275280879897077 | 1.503087927945232 | 0.051441666547930254 | 0.24276234757886103 | 3.30e-14 |
| [4,4] | 300.34910799781636 | 0.07587229468612591 | 0.8387929105132261 | 0.016912080910760966 | 0.07586310512616315 | 3.12e-14 |
| [8,8] | 300.39071107127455 | 0.020186952805083463 | 0.431839145971736 | 0.004785957566934641 | 0.020182358618652193 | 3.30e-14 |

Steady L2 order [4]->[8]: 1.91. Exact `total_energy` is 300 + 4/pi^2 = 300.4053.

### D3 `linear_elasticity` (`17-linear-elasticity`) -- ran live, satisfied

| level | displacement_magnitude_sq | strain_energy | l2_error | h1_seminorm_error | nodal_l2_error | forcing_defect |
|---|---|---|---|---|---|---|
| [2,2,2] | 0.0920927057583544 | 3.4534764659382917 | 0.38122970551471613 | 2.664964486717604 | 0.041655727456543425 | 6.9e-15 |
| [4,4,4] | 0.26413378924800385 | 7.761515637097678 | 0.12035499619448443 | 1.5994697573360508 | 0.007787582192893432 | 7.0e-15 |

`tests/test_linear_elasticity.py` additionally runs [8,8,8] and asserts L2 order > 1.7 and
H1 order > 0.8 between [4]^3 and [8]^3.

### D4 `stokes` (`25-stokes`) -- ran live, satisfied

| level | dissipation | mass_defect | divergence_l2_norm | velocity_l2_norm | pressure_l2_norm | solution_rms |
|---|---|---|---|---|---|---|
| [2,2] | 1.914828431372555e-4 | -5.4e-20 | 3.6764705882352997e-3 | 1.3141918253386782e-3 | 0.0430247425584654 | 0.03259535488026919 |
| [4,4] | 2.2066421652632795e-4 | -5.4e-20 | 1.4445654496759094e-3 | 1.4855671531276628e-3 | 0.04370162765293138 | 0.024414027011076814 |
| [8,8] | 2.2816088750453578e-4 | 1.8e-19 | 4.148014403573309e-4 | 1.5197216024530032e-3 | 0.04337468532947387 | 0.0192999018587831 |

**Cross-code agreement:** Sinbad's C11.20 record for the identical 59-dof [2,2] system is a
solution RMS of 3.2595e-2; the adapter's `solution_rms` (same zero-arithmetic-mean pressure
gauge) is 0.0325954 -- agreement to Sinbad's own printed precision.

### D5 `mixed_darcy` (`13-mixed-darcy`) -- ran live, refused (typed), with a finding

The declared problem (RT0/P0, `neumann flux = 0` on every wall, `div(flux) = 1`) has no
solution: `integral(source_term) = 1` but the impermeable boundary forces
`integral_boundary(flux . n) = 0`. The adapter computes that defect and returns
`unsupported_case` with it ("compatibility defect 1"); the recorded refusal is in
`tests/fixtures/recorded/mixed_darcy/2x2/result.json`.

Diagnosis of what Sinbad's runner solves instead (C11.21 reports `Completed`, full rank, no
pressure nullspace, `pressure_mean = 2.5948e-2`): dropping the pressure boundary integral, as
Sinbad's compiled `neumann flux = 0` term does, is the natural condition **p = 0 on the walls**,
not an impermeable wall. The adapter's `pressure_dirichlet_zero` variant (Python API only, not
a protocol capability) on the same [2,2,2] mesh gives `pressure_mean = 0.0259478` and
`total_flow = 1.0` (the unit source leaves through the walls). This is the SV0 Wave D
"mathematically wrong model converges internally but fails the independent oracle" pattern.

The solver itself is verified on a compatible manufactured problem
(p = cos(pi x) cos(pi y) cos(pi z), u = -grad p, f = 3 pi^2 p; impermeable walls satisfied
exactly): pressure L2 error 0.1799 / 0.0961 / 0.0488 and flux L2 error 0.9593 / 0.4981 /
0.2512 at [2]^3 / [4]^3 / [8]^3 (orders 0.98 and 0.99), system dimension 168 at [2]^3 as in
Sinbad's realization.

## Checks (2026-09-01, after the last source change)

| where | command | result |
|---|---|---|
| host (Python 3.14, no dolfinx) | `PYTHONPATH=src python3 -m pytest tests` | 68 passed, 6 skipped (the six dolfinx-gated modules) |
| host | `ruff check src tests scripts`; `ruff format --check src tests scripts` | clean |
| `dolfinx/dolfinx:stable` | `scripts/dolfinx-image.sh python3 -m pytest tests` | 100 passed, 1 skipped (sibling `sinbad/cases` mirror check; not mounted in the image) |
| `dolfinx/dolfinx:stable` | `scripts/record_live_runs.py` | 12 requests (11 satisfied, 1 typed refusal), all inside Sinbad's 30 s per-invocation budget |

Not run: Sinbad's own `tests/sv0_c5_fenicsx_oracle.rs` docker-gated test (Sinbad repository,
another lane's writer); the Poisson numbers it consumes are unchanged, so it is expected to
pass as before.

## Honestly open

- **Sinbad-side comparison for D2-D5 does not exist yet.** `run_independent_poisson_comparison`
  and `OracleCapability` are Poisson-only; the observables above are ready for a generic
  comparison keyed by observable id, but nothing on the Sinbad side consumes them (cross-repo
  needs below).
- `mixed_darcy` cannot produce a numeric comparison for `13-mixed-darcy` until the case is
  made consistent (a zero-mean source, or a pressure boundary condition declared as such).
- 3-D ladders are addressable only isotropically through the protocol's `[nx, ny]`.
- `linear_elasticity` and `mixed_darcy` in 3-D at [8]^3 and beyond take a few seconds of solve
  time; the first invocation in a fresh container additionally pays 7-15 s of JIT.
- `nonlinear_heat` reports final-time observables of a fixed 8-step BDF2 march; no temporal
  self-convergence observable is defined (the exact solution is steady, so none is meaningful
  for this case).

## Cross-repo needs (Sinbad; exact)

1. `sinbad/src/oracle.rs`: `OracleCapability` gains `NonlinearHeat`, `LinearElasticity`,
   `Stokes`, `MixedDarcy` (wire: `nonlinear_heat`, `linear_elasticity`, `stokes`,
   `mixed_darcy`), or is replaced by the GX-E5 "capability keyed by support profile" string.
   Without it Sinbad cannot even serialize a request for D2-D5.
2. `sinbad/src/oracle.rs`: `OracleRequest.refinement: [usize; 2]` -> `Vec<usize>` in a
   `sinbad-oracle-protocol/2`, so 3-D ladders are addressed exactly.
3. `sinbad/src/independent_comparison.rs`: generalize `run_independent_poisson_comparison`
   to a capability + observable-id-keyed comparison (energy-like observable at the finest
   shared level with a relative tolerance; an error-like observable ladder for the order
   floor), so `ObservableAgreement` comparisons bind for D2-D4 with no per-physics Rust.
   Suggested pairs: `nonlinear_heat` (`total_energy`, `steady_l2_error`),
   `linear_elasticity` (`displacement_magnitude_sq` or `strain_energy`, `l2_error`), `stokes`
   (`dissipation` or `solution_rms` at [2,2]; no error ladder, no `@mms`).
4. `sinbad/cases/13-mixed-darcy.toml` / `physics/corpus/13-mixed-darcy.res`: the declared
   problem is inconsistent (above). Either bind a zero-mean `source/source_term` or declare the
   wall condition as the pressure (natural) condition the runner actually realizes.
5. Optional: Sinbad's `OracleRun` evidence could reference the adapter-side manifest's
   `package_digest` and mesh/solution sha256 so the retained `.npz` files are part of the
   evidence graph.

## Deviations from the design docs

- ARCHITECTURE.md/GX-CONTRACTS: none in kind; this repository implements SV0-C5 as the SV0
  file describes it (adapter outside the Rust graph, protocol owned by Sinbad).
- SV0-D2 in the SV0 table is "transient diffusion"; this lane's brief numbered D2-D5 as
  nonlinear heat, elasticity, Stokes, Darcy and that ordering is used here.
- The refusal path for an inconsistent declared problem is an adapter decision not spelled out
  by the protocol: the protocol has no "inconsistent case" class, so `unsupported_case` carries
  the computed defect in its message.

## History

- 2026-08-31 `dfc66b7`, `70c5213`: protocol layer, honest unavailable path, first live Poisson.
- 2026-09-01: this entry.
