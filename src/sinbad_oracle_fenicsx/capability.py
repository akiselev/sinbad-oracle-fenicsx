"""Environment capability probe for the dolfinx-backed solve module.

Import of `dolfinx` (and its `mpi4py`/`petsc4py`/`ufl` dependencies) stays
lazy and confined to this module and `poisson.py`: the protocol layer
(`protocol.py`, `adapter.py`) must remain importable and testable with a
plain Python standard library, no network, and no dolfinx installation, so
ordinary offline CI can exercise every non-solve refusal path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DolfinxAvailability:
    available: bool
    version: str | None
    reason: str | None


def probe_dolfinx() -> DolfinxAvailability:
    """Attempts to import dolfinx and reports the outcome honestly.

    Never raises; a failed import is a normal, expected outcome on a machine
    without a FEniCSx environment and must translate to a typed refusal, not
    a crash and never a fabricated result.
    """
    try:
        import dolfinx  # noqa: F401
    except Exception as error:  # pragma: no cover - exercised when dolfinx is absent
        return DolfinxAvailability(available=False, version=None, reason=str(error))
    version = getattr(dolfinx, "__version__", None)
    if not version:
        return DolfinxAvailability(
            available=False,
            version=None,
            reason="dolfinx imported but reports no __version__",
        )
    return DolfinxAvailability(available=True, version=version, reason=None)
