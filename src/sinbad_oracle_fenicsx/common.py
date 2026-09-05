"""dolfinx helpers shared by every capability module.

Imports dolfinx and friends at module scope; only ever imported after
`capability.probe_dolfinx()` succeeded. Everything here is physics-neutral
(meshes, boundary dofs, quadrature, norms, evidence records); the physics
lives in the capability modules that mirror one Sinbad case each.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from .outcome import FieldRecord, MeshRecord, UnsupportedCase

COMM = MPI.COMM_WORLD


def subdivisions_for(dimension: int, refinement: tuple[int, ...]) -> tuple[int, ...]:
    """Maps the request's `refinement` onto the case's box ladder.

    `sinbad-oracle-request/2` carries `[nx, ny]` for a 2-D case and `[nx, ny, nz]` for a
    3-D one, read verbatim. A 3-D case still accepts the `/1` two-entry form `[n, n]` as the
    isotropic `[n, n, n]` of its Sinbad case ladder; an anisotropic two-entry request is
    refused rather than guessed, and a three-entry request cannot address a 2-D case.
    """
    if any(entry < 1 for entry in refinement):
        raise UnsupportedCase(
            f"refinement subdivisions must be positive, got {list(refinement)}"
        )
    if dimension == 2:
        if len(refinement) != 2:
            raise UnsupportedCase(
                f"refinement {list(refinement)} cannot address a 2-D case; expected [nx, ny]"
            )
        return tuple(refinement)
    if dimension == 3:
        if len(refinement) == 3:
            return tuple(refinement)
        nx, ny = refinement
        if nx != ny:
            raise UnsupportedCase(
                f"refinement {list(refinement)} cannot address a 3-D case through a two-entry "
                "refinement; send [nx, ny, nz] under sinbad-oracle-request/2, or an isotropic "
                "[n, n] (read as [n, n, n])"
            )
        return (nx, nx, nx)
    raise UnsupportedCase(f"unsupported case dimension {dimension}")


def unit_box(subdivisions: tuple[int, ...]) -> dmesh.Mesh:
    """Structured simplex mesh of the unit box, matching Finitum's `simplex_box`.

    2-D: dolfinx's default `right` diagonal splits each brick along the
    lower-left/upper-right diagonal, exactly Finitum's
    `build_quad_triangle_grid`. 3-D: six tetrahedra per brick sharing the
    main diagonal (Kuhn), matching `build_brick_tet_grid`.
    """
    if len(subdivisions) == 2:
        nx, ny = subdivisions
        return dmesh.create_unit_square(COMM, nx, ny, dmesh.CellType.triangle)
    nx, ny, nz = subdivisions
    return dmesh.create_unit_cube(COMM, nx, ny, nz, dmesh.CellType.tetrahedron)


def integral(domain: dmesh.Mesh, expr) -> float:
    local = fem.assemble_scalar(fem.form(expr))
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def volume(domain: dmesh.Mesh) -> float:
    return integral(domain, fem.Constant(domain, 1.0) * ufl.dx(domain=domain))


def l2_norm(domain: dmesh.Mesh, expr) -> float:
    return float(np.sqrt(max(integral(domain, ufl.inner(expr, expr) * ufl.dx), 0.0)))


def h1_seminorm(domain: dmesh.Mesh, expr) -> float:
    return l2_norm(domain, ufl.grad(expr))


def interpolation_points(space: fem.FunctionSpace):
    # dolfinx <= 0.8 exposes interpolation_points() as a method; newer releases (including
    # the current dolfinx/dolfinx:stable image, 0.11) make it a property. Accept both.
    points = space.element.interpolation_points
    return points() if callable(points) else points


def interpolate(space: fem.FunctionSpace, ufl_expr) -> fem.Function:
    function = fem.Function(space)
    function.interpolate(fem.Expression(ufl_expr, interpolation_points(space)))
    return function


def exterior_facets(domain: dmesh.Mesh):
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    return dmesh.exterior_facet_indices(domain.topology)


def dirichlet_everywhere(space: fem.FunctionSpace, value: fem.Function) -> fem.DirichletBC:
    """Dirichlet `value` on the whole boundary (the corpus's `box_faces = ["all"]` region)."""
    domain = space.mesh
    facets = exterior_facets(domain)
    dofs = fem.locate_dofs_topological(space, domain.topology.dim - 1, facets)
    return fem.dirichletbc(value, dofs)


def dirichlet_everywhere_sub(
    mixed_space: fem.FunctionSpace, sub: int, value: fem.Function
) -> fem.DirichletBC:
    """As `dirichlet_everywhere`, on sub-space `sub` of a mixed space."""
    domain = mixed_space.mesh
    facets = exterior_facets(domain)
    collapsed, _ = mixed_space.sub(sub).collapse()
    dofs = fem.locate_dofs_topological(
        (mixed_space.sub(sub), collapsed), domain.topology.dim - 1, facets
    )
    return fem.dirichletbc(value, dofs, mixed_space.sub(sub))


LU_OPTIONS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def linear_problem(a, ell, bcs, prefix: str, options: dict | None = None, **kwargs):
    """`LinearProblem` across the 0.7..0.11 constructor signatures (direct LU/MUMPS solve)."""
    petsc_options = dict(LU_OPTIONS if options is None else options)
    try:
        return LinearProblem(
            a, ell, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix=prefix, **kwargs
        )
    except TypeError:
        return LinearProblem(a, ell, bcs=bcs, petsc_options=petsc_options, **kwargs)


def solved(problem) -> fem.Function:
    """`LinearProblem.solve()` returns the function (<= 0.9) or a function/tuple (0.10+)."""
    result = problem.solve()
    if isinstance(result, tuple):
        return result[0]
    return result


def nodal_rms_error(function: fem.Function, exact: Callable[[np.ndarray], np.ndarray]) -> float:
    """Sinbad's `nodal_l2_error`: RMS of (dof value - exact) over every node and component.

    Valid for Lagrange spaces whose dofs are point evaluations at the nodes (P1 here, whose
    nodes are exactly the mesh vertices, so this matches Sinbad's per-vertex normalization).
    """
    space = function.function_space
    block_size = space.dofmap.index_map_bs
    coordinates = space.tabulate_dof_coordinates()[:, : space.mesh.geometry.dim]
    expected = np.asarray(exact(coordinates.T), dtype=np.float64)
    if expected.ndim == 1:
        expected = expected.reshape(1, -1)
    actual = function.x.array.reshape(-1, block_size)[: coordinates.shape[0]].T
    diff = actual - expected
    return float(np.sqrt(np.mean(diff**2)))


def mesh_record(domain: dmesh.Mesh, subdivisions: tuple[int, ...]) -> MeshRecord:
    gdim = domain.geometry.dim
    cell_type = domain.topology.cell_name()
    geometry = np.array(domain.geometry.x[:, :gdim], dtype=np.float64)
    # dolfinx 0.11 deprecates `geometry.dofmap` in favour of `geometry.dofmaps[0]`.
    dofmaps = getattr(domain.geometry, "dofmaps", None)
    dofmap = dofmaps[0] if dofmaps is not None else domain.geometry.dofmap
    topology = np.array(dofmap, dtype=np.int64).reshape(len(dofmap), -1)
    return MeshRecord(
        cell_type=str(cell_type),
        dimension=gdim,
        subdivisions=tuple(int(n) for n in subdivisions),
        geometry=geometry,
        topology=topology,
    )


def field_record(
    name: str, function: fem.Function, space: str, point_dofs: bool
) -> FieldRecord:
    fs = function.function_space
    components = fs.dofmap.index_map_bs
    coordinates = None
    if point_dofs:
        coordinates = np.array(
            fs.tabulate_dof_coordinates()[:, : fs.mesh.geometry.dim], dtype=np.float64
        )
    return FieldRecord(
        name=name,
        space=space,
        components=components,
        values=np.array(function.x.array, dtype=np.float64),
        dof_coordinates=coordinates,
    )
