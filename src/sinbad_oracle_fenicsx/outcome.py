"""Solve-module outcome contract shared by every capability module.

Pure standard library: the adapter and the offline tests import this without
dolfinx. A capability module (`poisson.py`, `nonlinear_heat.py`, ...) returns
one `SolveOutcome` carrying every normalized observable it defines plus the
raw evidence the adapter retains next to the result file (`evidence.py`):
the mesh it actually solved on and the discrete solution it actually
computed. A module that determines, from the request alone, that the declared
case has no solution it can honestly report raises `UnsupportedCase`; the
adapter turns that into a typed `unsupported_case` refusal, never a crash and
never a fabricated number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class UnsupportedCase(Exception):
    """The declared case cannot be independently solved as requested.

    Carries the exact, human-readable reason that becomes the refusal
    message on the wire (`OracleStatus::Refused { class: unsupported_case }`).
    """


@dataclass(frozen=True)
class MeshRecord:
    """The concrete mesh a capability module solved on."""

    cell_type: str
    dimension: int
    subdivisions: tuple[int, ...]
    #: shape (vertex_count, dimension)
    geometry: Any
    #: shape (cell_count, vertices_per_cell), indices into `geometry`
    topology: Any


@dataclass(frozen=True)
class FieldRecord:
    """One discrete solution field, exactly as the solver left it."""

    name: str
    #: space family and order as authored by the mirrored `.res` field
    space: str
    components: int
    #: shape (dof_count * components,), dolfinx interleaved-component layout
    values: Any
    #: shape (dof_count, dimension) when the space has point-evaluation dofs, else None
    dof_coordinates: Any | None = None


@dataclass(frozen=True)
class SolveOutcome:
    observables: Mapping[str, float]
    mesh: MeshRecord
    fields: tuple[FieldRecord, ...]
    #: solver-side facts worth retaining (iteration counts, steps, options); JSON-serializable
    notes: Mapping[str, Any] = field(default_factory=dict)
