"""dolfinx-backed independent solve of Sinbad's 13-mixed-darcy case (SV0-D5).

Mirrors `sinbad/cases/13-mixed-darcy.toml` and
`sinbad/physics/corpus/13-mixed-darcy.res`:

* domain: unit cube; `flux: vector(3) HDiv(order=0)` (lowest-order
  Raviart-Thomas, basix "RT" degree 1) and `pressure: scalar L2(order=0)`
  (P0), the `@inf_sup(pair = "RT0-P0")` pair;
* darcy_law: mobility_inverse * u + grad(p) = body_force with
  mobility_inverse = viscosity * inverse(permeability) = 1 and body_force = 0;
* mass_balance: div(u) = source_term, source_term = 1 (uniform);
* boundary `impermeable` on every wall: `neumann flux = 0`, i.e. the
  normal flux u . n = 0 on the whole boundary. In the mixed (H(div)) setting
  that is an ESSENTIAL condition on the flux space, imposed here on the RT0
  facet dofs; the pressure is then determined only up to a constant
  (zero-mean gauge, constant mode handed to the direct solver as the
  operator's nullspace).

Compatibility. With u . n = 0 everywhere, the divergence theorem forces
integral(source_term) = integral_boundary(u . n) = 0. The case binds a
uniform unit source, so integral(source_term) = 1 != 0: the declared problem
has NO solution. This module computes that defect first and refuses the
request (`unsupported_case`) with the defect in the message rather than
returning any number for it -- an independent oracle must not manufacture a
solution to an inconsistent problem, and this refusal is itself the
verification finding (Sinbad's own runner reaches `Completed` on this case
because its compiled boundary term drops the pressure boundary integral,
which is the natural condition p = 0 on the walls, not an impermeable wall;
see STATUS.md). The same solver runs unchanged on any compatible source
(`solve_with(...)`), which is how its own tests verify it against a
manufactured solution, and `pressure_dirichlet_zero` exposes the
p = 0-on-the-boundary variant for diagnosing what Sinbad actually solved.
"""

from __future__ import annotations

from typing import Callable

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from . import common
from .outcome import SolveOutcome, UnsupportedCase

CAPABILITY = "mixed_darcy"

MOBILITY_INVERSE = 1.0
CASE_SOURCE_TERM = 1.0
# Relative to integral(|source_term|); loose enough for the quadrature error of a smooth
# non-polynomial source, orders of magnitude below any genuine incompatibility.
COMPATIBILITY_TOLERANCE = 1e-8

MUMPS_NULLSPACE_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_24": 1,
    "mat_mumps_icntl_25": 0,
}


def _spaces(domain):
    rt0 = basix.ufl.element("RT", domain.basix_cell(), 1)
    p0 = basix.ufl.element("DG", domain.basix_cell(), 0)
    return fem.functionspace(domain, basix.ufl.mixed_element([rt0, p0]))


def solve(refinement: tuple[int, int]) -> SolveOutcome:
    """The declared 13-mixed-darcy case: uniform unit source, impermeable walls."""
    return solve_with(
        refinement, source_term=lambda x: CASE_SOURCE_TERM, boundary="impermeable"
    )


def solve_with(
    refinement: tuple[int, int],
    *,
    source_term: Callable,
    boundary: str = "impermeable",
    body_force: Callable | None = None,
    exact_pressure: Callable | None = None,
    exact_flux: Callable | None = None,
) -> SolveOutcome:
    """RT0-P0 mixed Darcy on the unit cube with `source_term(x)` (a UFL expression of
    `SpatialCoordinate`).

    `boundary`: `"impermeable"` imposes u . n = 0 essentially (the declared case);
    `"pressure_dirichlet_zero"` drops the boundary pressure term instead (natural p = 0),
    the full-rank variant kept only for diagnosing Sinbad's compiled boundary term.
    `exact_pressure`/`exact_flux` (UFL expressions of `SpatialCoordinate`) add
    `pressure_l2_error`/`flux_l2_error` to the outcome notes for the manufactured tests.
    """
    if boundary not in ("impermeable", "pressure_dirichlet_zero"):
        raise ValueError(f"unknown boundary treatment {boundary!r}")
    subdivisions = common.subdivisions_for(3, refinement)
    domain = common.unit_box(subdivisions)
    w_space = _spaces(domain)
    x = ufl.SpatialCoordinate(domain)
    f = source_term(x)
    if isinstance(f, (int, float)):
        f = fem.Constant(domain, float(f))
    f_body = body_force(x) if body_force is not None else fem.Constant(domain, (0.0, 0.0, 0.0))
    volume = common.volume(domain)

    impermeable = boundary == "impermeable"
    source_integral = common.integral(domain, f * ufl.dx)
    prescribed_flux_integral = 0.0  # neumann flux = 0 on every wall
    compatibility_defect = source_integral - prescribed_flux_integral
    if impermeable:
        scale = max(common.integral(domain, abs(f) * ufl.dx), 1.0)
        if abs(compatibility_defect) > COMPATIBILITY_TOLERANCE * scale:
            raise UnsupportedCase(
                "the declared mixed Darcy problem has no solution: impermeable walls "
                "(flux . n = 0 on the whole boundary) force integral(source_term) = "
                "integral_boundary(flux . n) = 0, but integral(source_term) = "
                f"{source_integral:.12g} over the unit cube (compatibility defect "
                f"{compatibility_defect:.12g}); an independent oracle cannot report a solution "
                "to an inconsistent problem"
            )

    (u, p) = ufl.TrialFunctions(w_space)
    (v, q) = ufl.TestFunctions(w_space)
    a = (
        MOBILITY_INVERSE * ufl.inner(u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    ell = ufl.inner(f_body, v) * ufl.dx - f * q * ufl.dx

    bcs = []
    if impermeable:
        u_space, _ = w_space.sub(0).collapse()
        zero_flux = fem.Function(u_space)
        zero_flux.x.array[:] = 0.0
        bcs.append(common.dirichlet_everywhere_sub(w_space, 0, zero_flux))

    problem = common.linear_problem(
        a, ell, bcs, "sinbad_oracle_darcy_", options=MUMPS_NULLSPACE_OPTIONS
    )
    nullspace = None
    if impermeable:
        null_vec = problem.A.createVecLeft()
        null_vec.set(0.0)
        p_dofs = np.unique(w_space.sub(1).dofmap.list.flatten())
        null_vec.setValues(p_dofs.astype(np.int32), np.ones(len(p_dofs)))
        null_vec.assemble()
        null_vec.normalize()
        nullspace = PETSc.NullSpace().create(vectors=[null_vec])
        problem.A.setNullSpace(nullspace)
    wh = common.solved(problem)
    if nullspace is not None and not nullspace.test(problem.A):
        raise UnsupportedCase(
            "the assembled mixed Darcy operator does not annihilate the constant pressure mode"
        )

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    uh.name, ph.name = "flux", "pressure"
    pressure_mean = common.integral(domain, ph * ufl.dx) / volume
    if impermeable:
        ph.x.array[:] -= pressure_mean

    n = ufl.FacetNormal(domain)
    observables = {
        "flux_l2_norm": common.l2_norm(domain, uh),
        "pressure_l2_norm": common.l2_norm(domain, ph),
        "mass_residual_l2": common.l2_norm(domain, ufl.div(uh) - f),
        "total_flow": common.integral(domain, ufl.dot(uh, n) * ufl.ds),
        "source_compatibility_defect": compatibility_defect,
    }
    notes = {
        "linear_solver": MUMPS_NULLSPACE_OPTIONS,
        "boundary": boundary,
        "pressure_gauge": "zero mean" if impermeable else "none (natural p = 0 on the walls)",
        "pressure_mean_before_gauge": pressure_mean,
        "mobility_inverse": MOBILITY_INVERSE,
        "system_dimension": int(
            w_space.dofmap.index_map.size_global * w_space.dofmap.index_map_bs
        ),
    }
    if exact_pressure is not None:
        notes["pressure_l2_error"] = common.l2_norm(domain, ph - exact_pressure(x))
    if exact_flux is not None:
        notes["flux_l2_error"] = common.l2_norm(domain, uh - exact_flux(x))
    return SolveOutcome(
        observables=observables,
        mesh=common.mesh_record(domain, subdivisions),
        fields=(
            common.field_record("flux", uh, "HDiv(order=0) RT0", point_dofs=False),
            common.field_record("pressure", ph, "L2(order=0) P0", point_dofs=False),
        ),
        notes=notes,
    )
