"""dolfinx-backed independent solve of Sinbad's 01-poisson manufactured case (SV0-D1).

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

import ufl
from dolfinx import fem

from . import common
from .outcome import SolveOutcome

CAPABILITY = "poisson"


def solve(refinement: tuple[int, ...]) -> SolveOutcome:
    """Solves the 01-poisson manufactured case at one [nx, ny] refinement."""
    subdivisions = common.subdivisions_for(2, refinement)
    domain = common.unit_box(subdivisions)
    v_space = fem.functionspace(domain, ("Lagrange", 1))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    k = fem.Constant(domain, 1.0)

    u_bc = common.interpolate(v_space, u_exact_ufl)
    bc = common.dirichlet_everywhere(v_space, u_bc)

    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    ell = f * v * ufl.dx
    uh = common.solved(common.linear_problem(a, ell, [bc], "sinbad_oracle_poisson_"))

    energy = common.integral(domain, 0.5 * k * ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
    error = uh - u_exact_ufl
    observables = {
        "energy": energy,
        "l2_error": common.l2_norm(domain, error),
        "h1_seminorm_error": common.h1_seminorm(domain, error),
    }
    return SolveOutcome(
        observables=observables,
        mesh=common.mesh_record(domain, subdivisions),
        fields=(common.field_record("u", uh, "H1(order=1)", point_dofs=True),),
        notes={"linear_solver": common.LU_OPTIONS},
    )


def solve_manufactured_poisson(refinement: tuple[int, ...]) -> dict[str, float]:
    """Observables only; kept for the existing tests and callers."""
    return dict(solve(refinement).observables)
