"""dolfinx-backed independent solve of Sinbad's 01-poisson manufactured case.

Mirrors `sinbad/cases/01-poisson.toml` and `sinbad/physics/corpus/01-poisson.res`
exactly, using dolfinx's own mesh, assembly, and linear solve -- never
Sinbad's numbers:

* domain: unit square [0, 1] x [0, 1];
* PDE: -div(k * grad(u)) = f, k = 1.0 constant diffusivity;
* manufactured solution: u_exact = sin(pi x) sin(pi y);
* forcing: f = 2 pi^2 sin(pi x) sin(pi y) (the exact forcing for the above;
  01-poisson.toml authors this directly as an expression binding rather than
  deriving it, see the binding comment there for why);
* boundary condition: Dirichlet u = u_exact on the full boundary (the
  "walls" region selects all box faces);
* discretization: P1 (order-1) Lagrange, matching
  `field u: unknown scalar H1(order=1) on Omega`;
* mesh family: `simplex_box`, i.e. a triangulated unit square, refined by an
  explicit [nx, ny] subdivision ladder -- `refinement` in the oracle request
  maps directly onto this ladder's [nx, ny] pair.

Only reached once `capability.probe_dolfinx()` has already confirmed dolfinx
is importable; this module imports dolfinx and friends at module scope and
will raise ImportError on import otherwise.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

SUPPORTED_OBSERVABLES = frozenset({"energy", "l2_error", "h1_seminorm_error"})


def solve_manufactured_poisson(refinement: tuple[int, int]) -> dict[str, float]:
    """Solves the 01-poisson manufactured case at one [nx, ny] refinement.

    Returns every observable in `SUPPORTED_OBSERVABLES`; the adapter selects
    the requested subset before reporting.
    """
    nx, ny = refinement
    if nx < 1 or ny < 1:
        raise ValueError(f"refinement subdivisions must be positive, got {refinement}")

    domain = dmesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dmesh.CellType.triangle)
    v_space = fem.functionspace(domain, ("Lagrange", 1))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    k = fem.Constant(domain, 1.0)

    # dolfinx <= 0.8 exposes interpolation_points() as a method; newer releases
    # (including the current dolfinx/dolfinx:stable image) make it a property
    # returning the ndarray directly. Accept both.
    interpolation_points = v_space.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    u_exact_expr = fem.Expression(u_exact_ufl, interpolation_points)
    u_bc = fem.Function(v_space)
    u_bc.interpolate(u_exact_expr)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(v_space, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    ell = f * v * ufl.dx

    problem_kwargs = {
        "bcs": [bc],
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"},
    }
    try:
        # Newer dolfinx (the current dolfinx/dolfinx:stable image) requires the
        # keyword-only petsc_options_prefix; older releases do not accept it.
        problem = LinearProblem(
            a, ell, petsc_options_prefix="sinbad_oracle_poisson_", **problem_kwargs
        )
    except TypeError:
        problem = LinearProblem(a, ell, **problem_kwargs)
    uh = problem.solve()

    def integral(expr) -> float:
        local = fem.assemble_scalar(fem.form(expr))
        return float(domain.comm.allreduce(local, op=MPI.SUM))

    energy = integral(0.5 * k * ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)

    error = uh - u_exact_ufl
    l2_error = float(np.sqrt(integral(ufl.inner(error, error) * ufl.dx)))
    h1_seminorm_error = float(
        np.sqrt(integral(ufl.inner(ufl.grad(error), ufl.grad(error)) * ufl.dx))
    )

    return {
        "energy": energy,
        "l2_error": l2_error,
        "h1_seminorm_error": h1_seminorm_error,
    }
