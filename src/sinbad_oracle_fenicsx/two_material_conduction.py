"""SC-W2 reference: assemble separate FEniCSx submesh operators, identify matching traces.

The unit cube has k=1 on x<1/2 and k=4 on x>1/2; the outer temperatures
are 400 K and 300 K. PETSc solves the assembled conforming union. No Sinbad
operator, mesh, field, or result is consumed.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc

from . import common
from .outcome import SolveOutcome, UnsupportedCase

CAPABILITY = "two_material_conduction"


def solve(refinement: tuple[int, ...]) -> SolveOutcome:
    subdivisions = common.subdivisions_for(3, refinement)
    parent = common.unit_box((2 * subdivisions[0], *subdivisions[1:]))
    spaces, coordinates, matrices, domains = [], [], [], []
    for side, coefficient in enumerate([1.0, 4.0]):
        cells = mesh.locate_entities(
            parent, 3, lambda x, s=side: x[0] <= 0.5 + 1e-12 if s == 0 else x[0] >= 0.5 - 1e-12
        )
        domain = mesh.create_submesh(parent, 3, cells)[0]
        space = fem.functionspace(domain, ("Lagrange", 1))
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        matrix = assemble_matrix(
            fem.form(coefficient * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
        )
        matrix.assemble()
        spaces.append(space)
        domains.append(domain)
        coordinates.append(space.tabulate_dof_coordinates())
        matrices.append(matrix)
    # Point DOFs carry roundoff from reference-to-physical maps even on binary
    # grids. Match uniquely within 1e-12 m, refusing an ambiguous correspondence.
    ids = {}
    mappings = []
    for points in coordinates:
        mapping = []
        for point in points:
            candidates = [
                dof
                for coordinate, dof in ids.items()
                if np.allclose(point, coordinate, rtol=0, atol=1e-12)
            ]
            if len(candidates) > 1:
                raise UnsupportedCase("ambiguous submesh trace correspondence")
            if candidates:
                mapping.append(candidates[0])
            else:
                key = tuple(float(x) for x in point)
                ids[key] = len(ids)
                mapping.append(ids[key])
        mappings.append(np.array(mapping, dtype=PETSc.IntType))
    expected_interface = (subdivisions[1] + 1) * (subdivisions[2] + 1)
    if len(set(mappings[0]) & set(mappings[1])) != expected_interface:
        raise UnsupportedCase("incomplete submesh trace coverage")
    union = PETSc.Mat().createAIJ([len(ids), len(ids)], nnz=40, comm=common.COMM)
    union.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    for matrix, mapping in zip(matrices, mappings):
        rows, columns, values = matrix.getValuesCSR()
        for row in range(len(mapping)):
            start, stop = rows[row], rows[row + 1]
            union.setValues(
                [mapping[row]],
                mapping[columns[start:stop]],
                values[start:stop],
                addv=PETSc.InsertMode.ADD,
            )
    union.assemble()
    rhs, solution = union.createVecRight(), union.createVecRight()
    prescribed = union.createVecRight()
    rhs.set(0)
    prescribed.set(0)
    boundary = []
    for point, dof in ids.items():
        if abs(point[0]) < 1e-12 or abs(point[0] - 1) < 1e-12:
            boundary.append(dof)
            prescribed.setValue(dof, 400.0 if point[0] < 0.5 else 300.0)
    prescribed.assemble()
    union.zeroRowsColumns(
        np.array(boundary, dtype=PETSc.IntType), diag=1.0, x=prescribed, b=rhs
    )
    solver = PETSc.KSP().create(common.COMM)
    solver.setOperators(union)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.solve(rhs, solution)
    if solver.getConvergedReason() <= 0:
        raise UnsupportedCase("submesh union linear solve did not converge")
    fields, flows, temperatures = [], [], []
    submeshes = []
    for side, (space, mapping, domain, coefficient) in enumerate(
        zip(spaces, mappings, domains, [1.0, 4.0])
    ):
        field = fem.Function(space)
        field.x.array[:] = solution.array[mapping]
        fields.append(common.field_record(f"T_{side}", field, "H1(order=1)", point_dofs=True))
        # Each half has volume 1/2: average axial flux equals twice its volume integral.
        flows.append(2.0 * common.integral(domain, -coefficient * ufl.grad(field)[0] * ufl.dx))
        mask = np.isclose(coordinates[side][:, 0], 0.5)
        temperatures.extend(field.x.array[mask].tolist())
        submeshes.append(
            {
                "geometry": domain.geometry.x.tolist(),
                "cells": int(domain.topology.index_map(3).size_global),
            }
        )
    return SolveOutcome(
        observables={
            "interface_temperature": sum(temperatures) / len(temperatures),
            "heat_flow": sum(flows) / 2,
        },
        mesh=common.mesh_record(parent, (2 * subdivisions[0], *subdivisions[1:])),
        fields=tuple(fields),
        notes={
            "submeshes": submeshes,
            "component_heat_flow": flows,
            "submesh_dof_to_union": [m.tolist() for m in mappings],
            "method": "separate submesh stiffness matrices with matching trace identification",
        },
    )
