"""Raw-evidence retention next to the result file.

SV0-C5's contract for an independent oracle is that a reported number is
never the only thing that survives: the mesh the adapter solved on, the
discrete solution it computed, the exact extraction script that turned that
solution into observables, and the exact upstream tool version all stay
retrievable so a comparison can be re-derived or disputed later. Sinbad's
harness already hashes the adapter's stdout/stderr/result bytes
(`sinbad/src/oracle.rs`); this module adds the adapter-side artifacts it
cannot see.

Layout, for a result written to `<dir>/<name>`:

    <dir>/<name>.evidence/
        manifest.json     sinbad-oracle-fenicsx-evidence/1 (below)
        mesh.npz          geometry (vertex_count x dim), topology (cell_count x nodes)
        solution.npz      one array per field (`<field>`), plus `<field>__dof_coordinates`
                          for point-dof spaces

The manifest carries: the request verbatim; the adapter identity; the full
toolchain (dolfinx version + git commit, basix, ufl, PETSc/petsc4py, mpi4py,
numpy, python); the extraction-script hashes (sha256 of every module that
touched the numbers, plus one digest over the whole package); sha256 of the
two `.npz` files; the observables as decimal floats (the wire carries bit
patterns); the normalization version and per-observable definitions; and the
solver notes the capability module recorded.

Only the standard library is imported at module scope; numpy is needed only
when writing (i.e. in a dolfinx environment).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .outcome import SolveOutcome
from .registry import CapabilitySpec

EVIDENCE_SCHEMA = "sinbad-oracle-fenicsx-evidence/1"
MANIFEST_NAME = "manifest.json"
MESH_NAME = "mesh.npz"
SOLUTION_NAME = "solution.npz"

PACKAGE_DIR = Path(__file__).resolve().parent


def evidence_dir_for(result_path: Path) -> Path:
    return result_path.with_name(result_path.name + ".evidence")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def package_script_hashes() -> dict[str, str]:
    """sha256 of every Python module in this package, keyed by module file name."""
    return {
        path.name: sha256_file(path)
        for path in sorted(PACKAGE_DIR.glob("*.py"))
        if path.name != "__pycache__"
    }


def package_digest(script_hashes: Mapping[str, str]) -> str:
    joined = "\n".join(f"{name} {digest}" for name, digest in sorted(script_hashes.items()))
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def write_evidence(
    result_path: Path,
    request: Mapping[str, Any],
    tool: Mapping[str, Any],
    spec: CapabilitySpec,
    outcome: SolveOutcome,
    toolchain: Mapping[str, Any],
    normalization_version: int,
) -> Path:
    """Writes the evidence directory for one satisfied solve; returns its path.

    Any existing directory at that path is replaced wholesale so a stale
    mesh from a previous invocation can never sit next to a fresh result.
    """
    import numpy as np

    directory = evidence_dir_for(result_path)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    mesh_path = directory / MESH_NAME
    np.savez(mesh_path, geometry=outcome.mesh.geometry, topology=outcome.mesh.topology)

    solution_arrays: dict[str, Any] = {}
    fields_manifest = []
    for record in outcome.fields:
        solution_arrays[record.name] = record.values
        entry = {
            "name": record.name,
            "space": record.space,
            "components": record.components,
            "dof_count": int(len(record.values)) // max(record.components, 1),
        }
        if record.dof_coordinates is not None:
            solution_arrays[f"{record.name}__dof_coordinates"] = record.dof_coordinates
            entry["dof_coordinates"] = f"{record.name}__dof_coordinates"
        fields_manifest.append(entry)
    solution_path = directory / SOLUTION_NAME
    np.savez(solution_path, **solution_arrays)

    script_hashes = package_script_hashes()
    extraction_modules = sorted(
        {f"{spec.module}.py", "common.py", "evidence.py", "adapter.py", "protocol.py"}
    )
    manifest = {
        "schema": EVIDENCE_SCHEMA,
        "request": dict(request),
        "tool": dict(tool),
        "capability": spec.capability,
        "sinbad_case": spec.sinbad_case,
        "model": spec.model,
        "toolchain": dict(toolchain),
        "python": sys.version,
        "extraction": {
            "modules": {name: script_hashes[name] for name in extraction_modules},
            "package_digest": package_digest(script_hashes),
            "package_scripts": script_hashes,
        },
        "mesh": {
            "file": MESH_NAME,
            "sha256": sha256_file(mesh_path),
            "cell_type": outcome.mesh.cell_type,
            "dimension": outcome.mesh.dimension,
            "subdivisions": list(outcome.mesh.subdivisions),
            "vertex_count": int(outcome.mesh.geometry.shape[0]),
            "cell_count": int(outcome.mesh.topology.shape[0]),
        },
        "solution": {
            "file": SOLUTION_NAME,
            "sha256": sha256_file(solution_path),
            "fields": fields_manifest,
        },
        "normalization_version": normalization_version,
        "normalization": dict(spec.normalization),
        "observables": {name: float(value) for name, value in outcome.observables.items()},
        "notes": _jsonable(outcome.notes),
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_manifest(result_path: Path) -> dict:
    return json.loads((evidence_dir_for(result_path) / MANIFEST_NAME).read_text("utf-8"))
