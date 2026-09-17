"""SHOW-2 independent mixed-P1 electrothermal component, SI units.

The default case is a 20 x 10 x 5 mm solid, 5 V across x, T=300 K at
z=0, with all other traces insulated. UFL differentiates the complete
coupled residual. This module never reads a Sinbad solution.
"""

from __future__ import annotations

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector
from petsc4py import PETSc

from . import common
from .outcome import SolveOutcome, UnsupportedCase

CAPABILITY = "electrothermal_component"


def solve(refinement: tuple[int, ...]) -> SolveOutcome:
    subdivisions = common.subdivisions_for(3, refinement)
    domain = common.unit_box(subdivisions)
    domain.geometry.x[:] *= np.array([0.02, 0.01, 0.005])
    element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    space = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))
    now, previous, older = (fem.Function(space) for _ in range(3))
    _, temperature_map = space.sub(1).collapse()
    temperature_map = np.asarray(temperature_map).reshape(-1)
    for state in (now, previous, older):
        state.x.array[temperature_map] = 300.0
    bcs = []
    cooled_dofs = None
    for sub, axis, coordinate, value in [
        (0, 0, 0.0, 5.0),
        (0, 0, 0.02, 0.0),
        (1, 2, 0.0, 300.0),
    ]:
        collapsed, _ = space.sub(sub).collapse()
        trace = fem.Function(collapsed)
        trace.x.array[:] = value
        facets = mesh.locate_entities_boundary(
            domain, 2, lambda x, a=axis, c=coordinate: np.isclose(x[a], c, atol=1e-12)
        )
        dofs = fem.locate_dofs_topological((space.sub(sub), collapsed), 2, facets)
        bcs.append(fem.dirichletbc(trace, dofs, space.sub(sub)))
        if sub == 1:
            cooled_dofs = dofs[0]
    potential, temperature = ufl.split(now)
    _, old_t = ufl.split(previous)
    _, older_t = ufl.split(older)
    electrical_test, thermal_test = ufl.TestFunctions(space)
    sigma = 100 / (1 + 0.004 * (temperature - 300))
    conductivity = 1 + 0.01 * (temperature - 300)
    power = sigma * ufl.dot(ufl.grad(potential), ufl.grad(potential))
    # Four-point tetrahedral quadrature, the declared Sinbad residual policy.
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    alpha = fem.Constant(domain, PETSc.ScalarType(1.0))
    beta = fem.Constant(domain, PETSc.ScalarType(-1.0))
    gamma = fem.Constant(domain, PETSc.ScalarType(0.0))
    rate = (alpha * temperature + beta * old_t + gamma * older_t) / 0.25
    residual = (
        sigma * ufl.dot(ufl.grad(potential), ufl.grad(electrical_test))
        + 1e6 * rate * thermal_test
        + conductivity * ufl.dot(ufl.grad(temperature), ufl.grad(thermal_test))
        - power * thermal_test
    ) * dx
    options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearProblem(
        residual,
        now,
        bcs=bcs,
        petsc_options=options,
        petsc_options_prefix="sinbad_oracle_electrothermal_",
    )
    trajectory = []
    for step in range(4):
        if step:
            alpha.value, beta.value, gamma.value = 1.5, -2.0, 0.5
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise UnsupportedCase("electrothermal Newton failed to converge")
        raw = assemble_vector(fem.form(residual))
        raw.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        cooling = -float(sum(raw.array[cooled_dofs]))
        storage = common.integral(domain, 1e6 * rate * dx)
        heating = common.integral(domain, power * dx)
        trajectory.append(
            {
                "time": (step + 1) * 0.25,
                "electrical_power": heating,
                "energy_rate": storage,
                "cooling_reaction": cooling,
                "balance_defect": storage - heating + cooling,
                "temperature_max": float(max(now.x.array[temperature_map])),
            }
        )
        raw.destroy()
        older.x.array[:] = previous.x.array
        previous.x.array[:] = now.x.array
    voltage = now.sub(0).collapse()
    temperature_field = now.sub(1).collapse()
    return SolveOutcome(
        observables={
            "electrical_power": common.integral(domain, power * dx),
            "thermal_energy": common.integral(domain, 1e6 * temperature * dx),
            "temperature_rise": common.integral(domain, (temperature - 300) * dx),
        },
        mesh=common.mesh_record(domain, subdivisions),
        fields=(
            common.field_record("V", voltage, "H1(order=1)", point_dofs=True),
            common.field_record("T", temperature_field, "H1(order=1)", point_dofs=True),
        ),
        notes={
            "trajectory": trajectory,
            "integrator": "bdf2",
            "startup": "bdf1",
            "step": 0.25,
            "final_time": 1.0,
            "newton_options": options,
            "residual_quadrature_degree": 2,
            "observable_quadrature_degree": 2,
        },
    )
