"""dolfinx-backed independent solve of Sinbad's 03-nonlinear-heat case (SV0-D2).

Mirrors `sinbad/cases/03-nonlinear-heat.toml` and
`sinbad/physics/corpus/03-nonlinear-heat.res`:

* domain: unit square, P1 Lagrange (`field T: state scalar H1(order=1)`);
* PDE: rho cp dT/dt - div(k(T) grad T) = Q with rho = cp = 1 and the
  field-dependent conductivity k(T) = 1 + 0.2 (T - 300);
* manufactured steady solution T_exact = 300 + sin(pi x) sin(pi y), used as
  the Dirichlet trace on every wall and as the initial condition
  (`initial_T` binds the same expression);
* forcing Q: the case authors the closed form by hand (Scientia cannot
  expand `div(k(T) grad T)` symbolically, contract C11.6). This module does
  NOT copy that closed form into the solve: it derives Q from the exact
  solution with UFL's own symbolic differentiation,
  Q = -div(k(T_exact) grad T_exact), and separately measures the L2 distance
  between the derived Q and the case-authored expression as the
  `forcing_defect` observable -- so a mis-derived case forcing is caught
  rather than silently reproduced;
* time policy: BDF2, fixed step 0.05 to final time 0.4 (eight steps), one
  BDF1 start-up step, Newton (PETSc SNES, direct LU) at every step. Because
  the exact solution is steady and the initial condition is exact, the
  trajectory should stay on the discrete steady state; the final-time
  observables are therefore comparable with a steady solve, and a direct
  steady Newton solve is reported alongside (`steady_l2_error`).
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem

from . import common
from .outcome import SolveOutcome, UnsupportedCase

CAPABILITY = "nonlinear_heat"

RHO = 1.0
CP = 1.0
T_REFERENCE = 300.0
K_SLOPE = 0.2
TIME_STEP = 0.05
FINAL_TIME = 0.4

NEWTON_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_rtol": 1e-12,
    "snes_atol": 1e-12,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def conductivity(temperature):
    return 1.0 + K_SLOPE * (temperature - T_REFERENCE)


def exact_temperature(x):
    return T_REFERENCE + ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def authored_forcing(x):
    """`source/Q` exactly as `03-nonlinear-heat.toml` binds it (for the defect check only)."""
    sx, sy = ufl.sin(ufl.pi * x[0]), ufl.sin(ufl.pi * x[1])
    cx, cy = ufl.cos(ufl.pi * x[0]), ufl.cos(ufl.pi * x[1])
    return 2.0 * ufl.pi**2 * (1.0 + K_SLOPE * sx * sy) * sx * sy - K_SLOPE * ufl.pi**2 * (
        cx**2 * sy**2 + sx**2 * cy**2
    )


def _newton_problem(residual, unknown, bcs, prefix: str) -> NonlinearProblem:
    try:
        return NonlinearProblem(
            residual,
            unknown,
            bcs=bcs,
            petsc_options=NEWTON_OPTIONS,
            petsc_options_prefix=prefix,
        )
    except TypeError:  # dolfinx <= 0.9: NewtonSolver-based API without SNES options
        return NonlinearProblem(residual, unknown, bcs=bcs)


def _solve_newton(problem: NonlinearProblem, unknown, label: str) -> int:
    """Runs one Newton solve to convergence; returns the iteration count."""
    solver = getattr(problem, "solver", None)
    if solver is not None:  # SNES-backed (dolfinx >= 0.10)
        problem.solve()
        reason = solver.getConvergedReason()
        iterations = int(solver.getIterationNumber())
        if reason <= 0:
            raise UnsupportedCase(
                f"Newton did not converge for {label} (SNES reason {reason}, "
                f"{iterations} iterations)"
            )
        return iterations
    from dolfinx.nls.petsc import NewtonSolver  # dolfinx <= 0.9

    newton = NewtonSolver(common.COMM, problem)
    newton.rtol = NEWTON_OPTIONS["snes_rtol"]
    newton.atol = NEWTON_OPTIONS["snes_atol"]
    newton.max_it = NEWTON_OPTIONS["snes_max_it"]
    iterations, converged = newton.solve(unknown)
    if not converged:
        raise UnsupportedCase(f"Newton did not converge for {label} ({iterations} iterations)")
    return int(iterations)


def solve(refinement: tuple[int, ...]) -> SolveOutcome:
    subdivisions = common.subdivisions_for(2, refinement)
    domain = common.unit_box(subdivisions)
    v_space = fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    t_exact = exact_temperature(x)

    q_derived = -ufl.div(conductivity(t_exact) * ufl.grad(t_exact))
    forcing_defect = common.l2_norm(domain, q_derived - authored_forcing(x))

    t_bc = common.interpolate(v_space, t_exact)
    bc = common.dirichlet_everywhere(v_space, t_bc)
    v = ufl.TestFunction(v_space)

    def diffusion(temperature):
        return (
            conductivity(temperature) * ufl.inner(ufl.grad(temperature), ufl.grad(v)) * ufl.dx
        )

    source = q_derived * v * ufl.dx

    # --- steady companion solve -------------------------------------------------------
    t_steady = fem.Function(v_space, name="T_steady")
    t_steady.interpolate(t_bc)
    steady_iterations = _solve_newton(
        _newton_problem(
            diffusion(t_steady) - source, t_steady, [bc], "sinbad_oracle_nlh_steady_"
        ),
        t_steady,
        "the steady nonlinear heat problem",
    )

    # --- BDF2 march, BDF1 start-up ----------------------------------------------------
    dt = fem.Constant(domain, TIME_STEP)
    t_now = fem.Function(v_space, name="T")
    t_prev = fem.Function(v_space)
    t_prev2 = fem.Function(v_space)
    t_now.interpolate(t_bc)
    t_prev.interpolate(t_bc)
    t_prev2.interpolate(t_bc)
    capacity = RHO * CP
    bdf1 = capacity * (t_now - t_prev) / dt * v * ufl.dx + diffusion(t_now) - source
    bdf2 = (
        capacity * (3.0 * t_now - 4.0 * t_prev + t_prev2) / (2.0 * dt) * v * ufl.dx
        + diffusion(t_now)
        - source
    )
    bdf1_problem = _newton_problem(bdf1, t_now, [bc], "sinbad_oracle_nlh_bdf1_")
    bdf2_problem = _newton_problem(bdf2, t_now, [bc], "sinbad_oracle_nlh_bdf2_")

    steps = int(round(FINAL_TIME / TIME_STEP))
    if abs(steps * TIME_STEP - FINAL_TIME) > 1e-12:
        raise UnsupportedCase("final_time must be an integer multiple of step")
    newton_iterations = []
    time = 0.0
    for step in range(steps):
        problem = bdf1_problem if step == 0 else bdf2_problem
        newton_iterations.append(
            _solve_newton(problem, t_now, f"BDF step {step + 1} (t = {time + TIME_STEP:.3f})")
        )
        time += TIME_STEP
        t_prev2.x.array[:] = t_prev.x.array
        t_prev.x.array[:] = t_now.x.array

    error = t_now - t_exact
    observables = {
        "total_energy": common.integral(domain, RHO * CP * t_now * ufl.dx),
        "l2_error": common.l2_norm(domain, error),
        "h1_seminorm_error": common.h1_seminorm(domain, error),
        "nodal_l2_error": common.nodal_rms_error(
            t_now, lambda p: T_REFERENCE + np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])
        ),
        "steady_l2_error": common.l2_norm(domain, t_steady - t_exact),
        "forcing_defect": forcing_defect,
    }
    return SolveOutcome(
        observables=observables,
        mesh=common.mesh_record(domain, subdivisions),
        fields=(
            common.field_record("T", t_now, "H1(order=1)", point_dofs=True),
            common.field_record("T_steady", t_steady, "H1(order=1)", point_dofs=True),
        ),
        notes={
            "integrator": "bdf2",
            "startup": "bdf1",
            "step": TIME_STEP,
            "final_time": time,
            "steps": steps,
            "newton_iterations_per_step": newton_iterations,
            "steady_newton_iterations": steady_iterations,
            "newton_options": NEWTON_OPTIONS,
        },
    )
