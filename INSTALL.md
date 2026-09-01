# Installing a working environment

## What this adapter needs

The protocol/CLI layer (`protocol.py`, `capability.py`, `adapter.py`) is
pure standard-library Python and needs nothing beyond Python 3.10+. Only
`poisson.py` (and the tests in `tests/test_poisson_manufactured.py`) need a
real FEniCSx stack:

- `dolfinx` (the Python bindings, `import dolfinx`), version 0.7 or later —
  `poisson.py` uses `fem.functionspace`, `fem.Expression`, `dolfinx.fem.petsc.LinearProblem`,
  `mesh.exterior_facet_indices`, all from that API generation.
- `mpi4py`
- `petsc4py` (dolfinx's `LinearProblem` is backed by PETSc)
- `ufl` (ships alongside dolfinx)
- `numpy`

## What was tried in this environment, and why it did not work

This adapter was scaffolded on a host with **no working dolfinx install**,
network access to PyPI/Docker Hub, but no conda/mamba and no root package
manager access. In order, what was checked (see
`lane-oracle-fenicsx.md` for the full trace):

1. `python3 -c "import dolfinx"` — `ModuleNotFoundError`.
2. `conda`/`mamba`/`micromamba` — none installed, and none can be installed
   without root (`pacman` needs root; `pip install` is
   externally-managed-environment blocked without `--break-system-packages`,
   which was deliberately not used for a system-wide install per the "no
   system-wide installs" constraint on this task).
3. `pip install fenics-dolfinx`, `pip install --dry-run fenics-dolfinx` —
   `fenics-dolfinx` is **not published on PyPI at all**
   (`https://pypi.org/pypi/fenics-dolfinx/json` returns 404). This is
   expected: upstream distributes dolfinx via conda-forge, Spack, apt (the
   FEniCS PPA), or Docker, not pip wheels, because it links PETSc/MPI native
   libraries pip cannot build portably.
4. `docker pull dolfinx/dolfinx:stable` — Docker was available and usable
   without root on this host, so this was attempted as the most promising
   path; see the lane report for the outcome (pull may still have been in
   progress when the rest of this repository was written, since a fresh
   pull of a scientific-computing image is multi-hundred-MB and can run past
   the time budget for environment setup).

## Recommended real environment

**conda-forge, when conda/mamba is available (preferred):**

```sh
conda create -n sinbad-oracle-fenicsx -c conda-forge \
  python=3.11 fenics-dolfinx mpich pyvista
conda activate sinbad-oracle-fenicsx
pip install -e /projects/sinbad/sinbad-oracle-fenicsx
pytest /projects/sinbad/sinbad-oracle-fenicsx/tests
```

**Docker (no conda needed), using the official image:**

```sh
docker run --rm -v /projects/sinbad/sinbad-oracle-fenicsx:/adapter \
  -w /adapter dolfinx/dolfinx:stable \
  bash -c "pip install -e . && pytest tests"
```

To actually run the adapter as Sinbad's harness would (as a plain
`command request-file result-file` executable), give the container the
adapter's console-script entrypoint:

```sh
docker run --rm -v /projects/sinbad/sinbad-oracle-fenicsx:/adapter \
  -w /adapter dolfinx/dolfinx:stable \
  bash -c "pip install -e . && sinbad-oracle-fenicsx request.json result.json"
```

A production deployment would more likely bake a pinned dolfinx image (exact
version recorded in `OracleToolIdentity.version`, per the protocol's own
"exact upstream tool version" requirement) rather than floating on `:stable`.

## Verifying an install works

```sh
python3 -c "import dolfinx; print(dolfinx.__version__)"
pytest tests/test_poisson_manufactured.py -v
```

The three convergence/correctness tests there should pass (not skip) once
dolfinx is importable; a skip there with dolfinx present indicates a broken
`mpi4py`/`petsc4py`/`ufl` sub-dependency, not a missing `dolfinx` package
itself.
