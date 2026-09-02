# sinbad-oracle-fenicsx

Independent finite-element verification oracle for
[Sinbad](../sinbad)'s SV0 trustworthy-simulation-factory (work package
SV0-C5, `sinbad/docs/simulation-vision/SV0-TRUSTWORTHY-SIMULATION-FACTORY.md`).

This adapter speaks Sinbad's sealed `sinbad-oracle-protocol/1` contract
(`sinbad/src/oracle.rs`) and independently recomputes results with
[FEniCSx/dolfinx](https://fenicsproject.org/) -- it never echoes Sinbad's own
numbers. Independence is the entire point: this is the adapter that makes
`IndependentlyVerified` reachable on the SV0 evidence ladder (contract
C11.18 sealed the first such claim, for Poisson, against this adapter). A
cross-code agreement is verification evidence, not physical validation, per
SV0's own non-goals.

## Scope (SV0-C5 D1-D5)

One capability per Sinbad case file; every capability implements the same
contract (`registry.py`, `outcome.py`, `evidence.py`): observables keyed by
observable id, one normalization version (`NORMALIZATION.md`), and raw
evidence retained next to the result file. Sinbad's comparison and
promotion path therefore needs no per-physics code on its side.

| capability | Sinbad case | model | what is solved | live status (dolfinx 0.11.0.post0) |
|---|---|---|---|---|
| `poisson` | `01-poisson` | `Poisson` | P1 manufactured Poisson, unit square | satisfied; reproduces the C11.18 numbers bit for bit |
| `nonlinear_heat` | `03-nonlinear-heat` | `NonlinearHeat` | P1, k(T) = 1 + 0.2 (T - 300), BDF2 0.05 to 0.4 with Newton per step, plus a steady Newton companion | satisfied |
| `linear_elasticity` | `17-linear-elasticity` | `LinearElasticity` | P1 vector, unit cube, lambda = 1.25, mu = 1, manufactured displacement | satisfied |
| `stokes` | `25-stokes` | `StokesFlow` | Taylor-Hood P2/P1, mu = 1.7, curl body force, no-slip cavity, zero-mean pressure | satisfied; reproduces Sinbad's C11.20 solution RMS to five digits |
| `mixed_darcy` | `13-mixed-darcy` | `MixedDarcy` | RT0/P0, impermeable walls, uniform unit source | **refused, typed**: the declared problem has no solution (see below) |

The recorded outputs of the live runs (request, result, evidence manifest
per ladder level) are in `tests/fixtures/recorded/` and are checked offline
by `tests/test_recorded_fixtures.py`. `STATUS.md` tabulates the numbers.

### Findings the oracle produced

- **Hand-derived case forcings are right.** `03-nonlinear-heat.toml`'s
  `source/Q` and `17-linear-elasticity.toml`'s `source/body_force` were
  authored by hand because Scientia cannot expand those shapes (contract
  C11.6). The adapter derives both from the exact solutions with UFL's own
  symbolic differentiation and reports the L2 distance as `forcing_defect`:
  3e-14 and 7e-15 respectively.
- **Sinbad's Stokes solution is independently confirmed.** On the identical
  `[2, 2]` Taylor-Hood mesh (59 dofs) the adapter's `solution_rms` is
  0.0325954 against Sinbad's recorded 3.2595e-2 (C11.20).
- **`13-mixed-darcy` as declared is inconsistent.** `neumann flux = 0` on
  every wall with `div(flux) = 1` violates the divergence theorem
  (`integral(source) = 1 != 0 = integral_boundary(flux . n)`), so no solution
  exists and the adapter refuses. Sinbad's runner reaches `Completed` on this
  case because its compiled boundary term drops the pressure boundary
  integral, which is the natural condition p = 0 on the walls, not an
  impermeable wall: the adapter's p = 0 variant reproduces Sinbad's recorded
  `pressure_mean` (0.0259478 vs 2.5948e-2, C11.21) on the same mesh. The
  RT0/P0 solver itself converges at first order on a compatible manufactured
  problem (`tests/test_mixed_darcy.py`).

## Protocol conformance

Invoked exactly as `sinbad-oracle-fenicsx <request-file> <result-file>`.
Requests/results follow `sinbad-oracle-protocol/1` verbatim; see
`src/sinbad_oracle_fenicsx/protocol.py` for the exact wire shapes this
mirrors from `sinbad/src/oracle.rs` and `sinbad/src/verification_execution.rs`
(including the `FiniteF64` bit-pattern encoding, which is easy to get wrong
from the Python side -- read that module's docstring before touching it).

Refusal behavior, in check order:

| Situation | Refusal class |
|---|---|
| requested tool identity != this adapter's actual identity | `version_lie` |
| capability not in `registry.CAPABILITIES` | `unsupported_case` |
| dolfinx not importable in this environment | `unsupported_case` |
| observable id(s) the capability does not define | `unsupported_case` |
| the capability module refuses (anisotropic 3-D refinement, a declared problem with no solution, Newton did not converge) | `unsupported_case` |
| bad arguments / unreadable / malformed request | non-zero exit, no result file (`crash`) |
| unexpected exception during solve, or evidence could not be retained | non-zero exit, no result file (`crash`) |

The frozen protocol has no distinct "unavailable" refusal class, so a
missing dolfinx install is reported honestly as `unsupported_case` -- never
a fabricated result.

Protocol limitations this adapter works within (recorded as cross-repo
needs in `STATUS.md`): `OracleCapability` on the Sinbad side declares only
`poisson` today, and `refinement` is a fixed `[nx, ny]`, so a 3-D case reads
`[n, n]` as `[n, n, n]` and refuses anything anisotropic.

## Raw evidence

A satisfied run writes `<result-file>.evidence/` beside the result file:
`manifest.json` (`sinbad-oracle-fenicsx-evidence/1`: the request, the
adapter identity, the full toolchain with the dolfinx git commit, sha256 of
every extraction script and of the package, the normalization definitions,
the observables as decimals, solver notes), `mesh.npz` (geometry, topology)
and `solution.npz` (every discrete field, plus dof coordinates for
point-dof spaces). Sinbad's harness already hashes stdout/stderr/result;
this is the adapter-side complement. A refusal retains nothing.

## Layout

```
src/sinbad_oracle_fenicsx/
  protocol.py           # sinbad-oracle-protocol/1 (de)serialization, no third-party deps
  registry.py           # capability table: case, model, module, observable normalization
  outcome.py            # SolveOutcome / UnsupportedCase contract between adapter and modules
  evidence.py           # raw-evidence sidecar writer (numpy only when writing)
  normalization_doc.py  # renders NORMALIZATION.md from the registry
  capability.py         # honest dolfinx-importable probe + toolchain identity
  adapter.py            # CLI entrypoint / refusal dispatch, no third-party deps
  common.py             # dolfinx helpers: meshes, bcs, quadrature, norms, records
  poisson.py nonlinear_heat.py linear_elasticity.py stokes.py mixed_darcy.py
tests/
  test_protocol.py test_capability.py test_registry.py test_evidence.py   # offline
  test_adapter_cli.py test_recorded_fixtures.py                            # offline
  test_poisson_manufactured.py test_nonlinear_heat.py test_linear_elasticity.py
  test_stokes.py test_mixed_darcy.py test_adapter_live.py                  # need dolfinx
  fixtures/recorded/<capability>/<nx>x<ny>/{request,result,manifest}.json
scripts/
  dolfinx-image.sh      # run anything inside the official image as your uid
  record_live_runs.py   # re-record tests/fixtures/recorded (inside the image)
```

## Running

Offline (plain Python 3.10+, numpy for the evidence tests, no network; this
is what ordinary CI runs, matching SV0's "license-free and
external-tool-free" gate):

```sh
PYTHONPATH=src python3 -m pytest tests
ruff check src tests scripts && ruff format --check src tests scripts
```

With dolfinx, through the official image (see `INSTALL.md`):

```sh
scripts/dolfinx-image.sh python3 -m pytest tests
scripts/dolfinx-image.sh python3 scripts/record_live_runs.py tests/fixtures/recorded
```

## License

Deferred, per workspace convention
(`sinbad/CLAUDE.md`: "Defer licensing and release packaging until a concrete
SV3 export or release-preparation work package requires them").
