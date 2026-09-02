"""Environment capability probe for the dolfinx-backed solve modules.

Import of `dolfinx` (and its `mpi4py`/`petsc4py`/`ufl` dependencies) stays
lazy and confined to this module, `common.py` and the capability modules:
the protocol layer (`protocol.py`, `adapter.py`, `registry.py`) must remain
importable and testable with a plain Python standard library, no network,
and no dolfinx installation, so ordinary offline CI can exercise every
non-solve refusal path.
"""

from __future__ import annotations

import platform
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


def toolchain_identity() -> dict[str, str]:
    """Exact versions of every library between the request and the numbers.

    Only called once `probe_dolfinx()` succeeded; each sub-dependency is
    still probed defensively so one missing optional attribute never turns a
    finished solve into a crash.
    """
    identity: dict[str, str] = {"python": platform.python_version()}

    def record(name: str, getter) -> None:
        try:
            identity[name] = str(getter())
        except Exception as error:  # pragma: no cover - environment-dependent
            identity[name] = f"unavailable: {error}"

    import dolfinx

    record("dolfinx", lambda: dolfinx.__version__)
    record("dolfinx_git_commit", lambda: dolfinx.git_commit_hash)
    record("basix", lambda: __import__("basix").__version__)
    record("ufl", lambda: __import__("ufl").__version__)
    record("numpy", lambda: __import__("numpy").__version__)
    record("mpi4py", lambda: __import__("mpi4py").__version__)
    record("petsc4py", lambda: __import__("petsc4py").__version__)

    def petsc_version() -> str:
        from petsc4py import PETSc

        return ".".join(str(part) for part in PETSc.Sys.getVersion())

    record("petsc", petsc_version)
    return identity
