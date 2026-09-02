# Installing a working environment

## What this adapter needs

The protocol/CLI layer (`protocol.py`, `registry.py`, `outcome.py`,
`capability.py`, `adapter.py`) is pure standard-library Python and needs
nothing beyond Python 3.10+; `evidence.py` needs numpy only when it writes.
The capability modules (`poisson.py`, `nonlinear_heat.py`,
`linear_elasticity.py`, `stokes.py`, `mixed_darcy.py`, and `common.py`) need
a real FEniCSx stack:

- `dolfinx` (the Python bindings) 0.10 or later for the SNES-backed
  `NonlinearProblem` the nonlinear-heat capability uses (`petsc_options`,
  `petsc_options_prefix`); the linear capabilities fall back to the 0.7-0.9
  constructor signatures;
- `basix` (mixed elements for Taylor-Hood and RT0/P0), `ufl`;
- `mpi4py`, `petsc4py` with MUMPS (every direct solve here uses MUMPS; the
  Stokes and Darcy saddle points rely on its null-pivot handling);
- `numpy`.

## The environment used for the recorded runs

This host has no native dolfinx (`fenics-dolfinx` is not on PyPI; upstream
ships conda-forge, Spack, apt, or Docker), no conda, and no root package
manager. Docker works without root, and the official image is present:

| | |
|---|---|
| image | `dolfinx/dolfinx:stable` |
| digest | `sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8` |
| dolfinx | `0.11.0.post0`, git `fefdb2201b80a8f59527de2d461b9056906661d8` |
| basix / ufl | `0.11.0` / `2026.1.0` |
| petsc4py / PETSc | `3.25.1` / `3.25.1` (real, 32-bit indices) |
| python / numpy | `3.12.3` / `2.4.6` |

`OracleToolIdentity.version` reports `0.11.0.post0` from that image; the
evidence manifest records the whole table above per run.

`scripts/dolfinx-image.sh <command...>` runs a command inside that image
with this checkout mounted at `/adapter`, `src` on `PYTHONPATH` (prepended
to the image's own, which is how the image locates dolfinx), your uid so
nothing written back is root-owned, and `IO_DIR` (if set) mounted at `/io`:

```sh
scripts/dolfinx-image.sh python3 -m pytest tests
IO_DIR=/some/dir scripts/dolfinx-image.sh python3 -m sinbad_oracle_fenicsx /io/request.json /io/result.json
```

Sinbad's own docker-gated test (`sinbad/tests/sv0_c5_fenicsx_oracle.rs`)
builds its own throwaway image from this checkout (`pip install -e`) and
fronts it with a wrapper script; it needs only
`SINBAD_FENICSX_ADAPTER_DIR=/projects/sinbad/sinbad-oracle-fenicsx` and a
working `docker`.

Each container start pays dolfinx's JIT compilation for the forms
(roughly 7-15 s per capability on this host); the actual solves at the
recorded ladders take well under a second. Sinbad's per-invocation
`ORACLE_TIMEOUT` is 30 s, which every recorded level met.

## Recommended real environment

**conda-forge, when conda/mamba is available:**

```sh
conda create -n sinbad-oracle-fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.11 mpich mumps
conda activate sinbad-oracle-fenicsx
pip install -e /projects/sinbad/sinbad-oracle-fenicsx
pytest /projects/sinbad/sinbad-oracle-fenicsx/tests
```

A production deployment should pin the image by digest (as above) rather
than float on `:stable`, so that `OracleToolIdentity.version` names one
exact upstream build as the protocol requires.

## Verifying an install works

```sh
python3 -c "import dolfinx; print(dolfinx.__version__)"
pytest tests/test_adapter_live.py -v
```

Every dolfinx-gated test should pass (not skip); a skip with dolfinx present
indicates a broken `mpi4py`/`petsc4py`/`basix`/`ufl` sub-dependency, not a
missing `dolfinx` package itself.
