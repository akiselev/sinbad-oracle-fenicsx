# sinbad-oracle-fenicsx

Independent finite-element verification oracle for
[Sinbad](../sinbad)'s SV0 trustworthy-simulation-factory (work package
SV0-C5, `docs/simulation-vision/SV0-TRUSTWORTHY-SIMULATION-FACTORY.md`).

This adapter speaks Sinbad's sealed `sinbad-oracle-protocol/1` contract
(`sinbad/src/oracle.rs`) and independently recomputes results with
[FEniCSx/dolfinx](https://fenicsproject.org/) — it never echoes Sinbad's own
numbers. Independence is the entire point: this is the first adapter that
can make `IndependentlyVerified` reachable on the SV0 evidence ladder
(ladder level 9, "independent implementation, literature, or experimental
comparison"). A cross-code agreement is verification evidence, not physical
validation, per SV0's own non-goals.

## Scope

- **Landed:** Poisson (`sinbad/cases/01-poisson.toml` /
  `physics/corpus/01-poisson.res`) — a manufactured-solution Dirichlet
  problem on the unit square, `-div(k grad(u)) = f`, `k = 1`,
  `u_exact = sin(pi x) sin(pi y)`, P1 Lagrange on a triangulated mesh
  refined by an explicit `[nx, ny]` ladder.
- **Deferred, per SV0-D2–D5:** transient diffusion, nonlinear heat, linear
  elasticity, Stokes. Each needs its own solve module and observable set;
  none is implemented here yet.

## Protocol conformance

Invoked exactly as `sinbad-oracle-fenicsx <request-file> <result-file>`.
Requests/results follow `sinbad-oracle-protocol/1` verbatim; see
`src/sinbad_oracle_fenicsx/protocol.py` for the exact wire shapes this
mirrors from `sinbad/src/oracle.rs` and `sinbad/src/verification_execution.rs`
(including the `FiniteF64` bit-pattern encoding, which is easy to get wrong
from the Python side — read that module's docstring before touching it).

Refusal behavior:

| Situation | Refusal class |
|---|---|
| requested tool identity != this adapter's actual identity | `version_lie` |
| unsupported `capability` | `unsupported_case` |
| dolfinx not importable in this environment | `unsupported_case` |
| unsupported observable name(s) requested | `unsupported_case` |
| bad arguments / unreadable / malformed request | non-zero exit, no result file (`crash`) |
| unexpected exception during solve | non-zero exit, no result file (`crash`) |

The frozen protocol has no distinct "unavailable" refusal class, so a
missing dolfinx install is reported honestly as `unsupported_case` — never
a fabricated result. See `src/sinbad_oracle_fenicsx/adapter.py`'s module
docstring for the exact ordering and reasoning, and `INSTALL.md` for what
this host currently has.

## Layout

```
src/sinbad_oracle_fenicsx/
  protocol.py    # sinbad-oracle-protocol/1 (de)serialization, no third-party deps
  capability.py  # honest dolfinx-importable probe, no third-party deps
  adapter.py     # CLI entrypoint / refusal dispatch, no third-party deps
  poisson.py     # dolfinx-backed solve of the 01-poisson manufactured case
tests/
  test_protocol.py             # offline
  test_capability.py           # offline
  test_adapter_cli.py          # offline, subprocess-based
  test_poisson_manufactured.py # requires dolfinx; skipped otherwise
```

The protocol/CLI layer has zero third-party dependencies by design, so
`pytest tests/test_protocol.py tests/test_capability.py tests/test_adapter_cli.py`
runs anywhere with a plain Python 3.10+ and no network — matching SV0's own
"ordinary CI remains license-free and external-tool-free" gate.

## License

Deferred, per workspace convention
(`sinbad/CLAUDE.md`: "Defer licensing and release packaging until a concrete
SV3 export or release-preparation work package requires them").
