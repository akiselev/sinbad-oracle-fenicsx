"""dolfinx-backed independent solve of Sinbad's 25-stokes case (SV0-D4).

Mirrors `sinbad/cases/25-stokes.toml` and `sinbad/physics/corpus/25-stokes.res`:

* domain: unit square; Taylor-Hood pair `velocity: vector(2) H1(order=2)`,
  `pressure: scalar L2(order=1)` -- P2/P1 continuous Lagrange
  (`@inf_sup(pair = "Taylor-Hood")`);
* momentum: -div(2 mu sym_grad(u)) + grad(p) = f, mu = 1.7 (rho is bound
  but does not enter the steady Stokes equations);
* incompressibility: div(u) = 0;
* body force f = [y - 0.5, 0], a pure shear with curl(f) = -1 so that the
  enclosed cavity must flow (see the case file's own explanation);
* no-slip u = 0 on every wall;
* the pressure is determined up to a constant on this fully enclosed
  domain: the constant mode is supplied to the direct solver as the
  operator's nullspace and the reported pressure is gauged to zero mean
  (Sinbad's own runner projects the same mode out, contract C11.20).

The weak form is written in its symmetric saddle-point arrangement
(the `-q div(u)` row sign is the `[system].equation_sign = { incompressibility
= -1.0 }` gauge the case file records); the solution is unaffected by that
row sign.
"""

from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from . import common
from .outcome import SolveOutcome, UnsupportedCase

CAPABILITY = "stokes"

VISCOSITY = 1.7

MUMPS_NULLSPACE_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # MUMPS: detect null pivots (24) and treat them as exact zeros (25 = 0) so the singular
    # (constant-pressure) saddle-point operator factors cleanly.
    "mat_mumps_icntl_24": 1,
    "mat_mumps_icntl_25": 0,
}


def strain_rate(u):
    return ufl.sym(ufl.grad(u))


def viscous_stress(u):
    return 2.0 * VISCOSITY * strain_rate(u)


def solve(refinement: tuple[int, int]) -> SolveOutcome:
    subdivisions = common.subdivisions_for(2, refinement)
    domain = common.unit_box(subdivisions)
    p2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    p1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    w_space = fem.functionspace(domain, basix.ufl.mixed_element([p2, p1]))
    x = ufl.SpatialCoordinate(domain)
    body_force = ufl.as_vector([x[1] - 0.5, 0.0])

    u_space, _ = w_space.sub(0).collapse()
    noslip = fem.Function(u_space)
    noslip.x.array[:] = 0.0
    bc = common.dirichlet_everywhere_sub(w_space, 0, noslip)

    (u, p) = ufl.TrialFunctions(w_space)
    (v, q) = ufl.TestFunctions(w_space)
    a = (
        ufl.inner(viscous_stress(u), strain_rate(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    ell = ufl.inner(body_force, v) * ufl.dx

    problem = common.linear_problem(
        a, ell, [bc], "sinbad_oracle_stokes_", options=MUMPS_NULLSPACE_OPTIONS
    )
    # Constant-pressure nullspace, attached to the operator before it is assembled/factored.
    null_vec = problem.A.createVecLeft()
    null_vec.set(0.0)
    p_dofs = w_space.sub(1).dofmap.list.flatten()
    p_dofs = np.unique(p_dofs)
    null_vec.setValues(p_dofs.astype(np.int32), np.ones(len(p_dofs)))
    null_vec.assemble()
    null_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[null_vec])
    problem.A.setNullSpace(nullspace)
    wh = common.solved(problem)
    if not nullspace.test(problem.A):
        raise UnsupportedCase(
            "the assembled Stokes operator does not annihilate the constant pressure mode"
        )

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    uh.name, ph.name = "velocity", "pressure"
    volume = common.volume(domain)
    ph.x.array[:] -= common.integral(domain, ph * ufl.dx) / volume

    # Sinbad's `pressure_mean` gauge is the arithmetic mean of the pressure block's dof values
    # (contract C11.19/C11.20), not the integral mean; `solution_rms` uses that same gauge so the
    # two codes' dof-level RMS values are like for like on an identical mesh.
    p_nodal_gauged = ph.x.array - np.mean(ph.x.array)
    all_dofs = np.concatenate([uh.x.array, p_nodal_gauged])
    observables = {
        "dissipation": common.integral(
            domain, ufl.inner(viscous_stress(uh), strain_rate(uh)) * ufl.dx
        ),
        "mass_defect": common.integral(domain, ufl.div(uh) * ufl.dx),
        "divergence_l2_norm": common.l2_norm(domain, ufl.div(uh)),
        "velocity_l2_norm": common.l2_norm(domain, uh),
        "pressure_l2_norm": common.l2_norm(domain, ph),
        "solution_rms": float(np.sqrt(np.mean(all_dofs**2))),
    }
    return SolveOutcome(
        observables=observables,
        mesh=common.mesh_record(domain, subdivisions),
        fields=(
            common.field_record("velocity", uh, "H1(order=2) vector(2)", point_dofs=True),
            common.field_record("pressure", ph, "L2(order=1) continuous", point_dofs=True),
        ),
        notes={
            "linear_solver": MUMPS_NULLSPACE_OPTIONS,
            "viscosity": VISCOSITY,
            "pressure_gauge": "zero mean",
            "system_dimension": int(len(all_dofs)),
        },
    )
