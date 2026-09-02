"""dolfinx-backed independent solve of Sinbad's 17-linear-elasticity case (SV0-D3).

Mirrors `sinbad/cases/17-linear-elasticity.toml` and
`sinbad/physics/corpus/17-linear-elasticity.res`:

* domain: unit cube, `field displacement: unknown vector(3) H1(order=1)` --
  P1 vector Lagrange on tetrahedra;
* isotropic law: stress = lambda tr(strain) I + 2 mu strain,
  strain = sym_grad(u), lambda = 1.25 Pa, mu = 1.0 Pa;
* PDE: -div(stress) = body_force;
* manufactured displacement u_exact = [phi, phi, phi],
  phi = sin(pi x) sin(pi y) sin(pi z), which vanishes on the whole boundary
  (`clamp` selects every face; Dirichlet u = u_exact there);
* body force: the case authors the closed form by hand. As for nonlinear
  heat, this module derives it with UFL from the exact displacement,
  f = -div(stress(sym_grad(u_exact))), and reports the L2 distance to the
  authored expression as `forcing_defect`;
* the case's ladder is isotropic `[[2,2,2],[4,4,4]]`; the protocol's
  two-entry refinement `[n, n]` is read as `[n, n, n]`.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from . import common
from .outcome import SolveOutcome

CAPABILITY = "linear_elasticity"

LAME_LAMBDA = 1.25
LAME_MU = 1.0


def strain(u):
    return ufl.sym(ufl.grad(u))


def stress(u):
    eps = strain(u)
    return LAME_LAMBDA * ufl.tr(eps) * ufl.Identity(3) + 2.0 * LAME_MU * eps


def exact_displacement(x):
    phi = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])
    return ufl.as_vector([phi, phi, phi])


def authored_body_force(x):
    """`source/body_force` exactly as `17-linear-elasticity.toml` binds it."""
    sx, sy, sz = (ufl.sin(ufl.pi * x[i]) for i in range(3))
    cx, cy, cz = (ufl.cos(ufl.pi * x[i]) for i in range(3))
    pi2 = ufl.pi**2
    return ufl.as_vector(
        [
            5.25 * pi2 * sx * sy * sz - 2.25 * pi2 * (cx * cy * sz + cx * cz * sy),
            5.25 * pi2 * sx * sy * sz - 2.25 * pi2 * (cx * cy * sz + cy * cz * sx),
            5.25 * pi2 * sx * sy * sz - 2.25 * pi2 * (cx * cz * sy + cy * cz * sx),
        ]
    )


def solve(refinement: tuple[int, int]) -> SolveOutcome:
    subdivisions = common.subdivisions_for(3, refinement)
    domain = common.unit_box(subdivisions)
    v_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_displacement(x)

    f_derived = -ufl.div(stress(u_exact))
    forcing_defect = common.l2_norm(domain, f_derived - authored_body_force(x))

    u_bc = common.interpolate(v_space, u_exact)
    bc = common.dirichlet_everywhere(v_space, u_bc)

    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    a = ufl.inner(stress(u), strain(v)) * ufl.dx
    ell = ufl.inner(f_derived, v) * ufl.dx
    uh = common.solved(common.linear_problem(a, ell, [bc], "sinbad_oracle_elasticity_"))

    error = uh - u_exact
    observables = {
        "displacement_magnitude_sq": common.integral(domain, ufl.dot(uh, uh) * ufl.dx),
        "strain_energy": common.integral(
            domain, 0.5 * ufl.inner(strain(uh), stress(uh)) * ufl.dx
        ),
        "l2_error": common.l2_norm(domain, error),
        "h1_seminorm_error": common.h1_seminorm(domain, error),
        "nodal_l2_error": common.nodal_rms_error(
            uh,
            lambda p: np.tile(
                np.sin(np.pi * p[0]) * np.sin(np.pi * p[1]) * np.sin(np.pi * p[2]), (3, 1)
            ),
        ),
        "forcing_defect": forcing_defect,
    }
    return SolveOutcome(
        observables=observables,
        mesh=common.mesh_record(domain, subdivisions),
        fields=(
            common.field_record("displacement", uh, "H1(order=1) vector(3)", point_dofs=True),
        ),
        notes={
            "linear_solver": common.LU_OPTIONS,
            "lame_lambda": LAME_LAMBDA,
            "lame_mu": LAME_MU,
        },
    )
