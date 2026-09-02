"""Offline tests for raw-evidence retention (needs numpy for the .npz files, not dolfinx)."""

import hashlib
import json

import pytest

np = pytest.importorskip("numpy")

from sinbad_oracle_fenicsx import evidence, registry  # noqa: E402
from sinbad_oracle_fenicsx.outcome import FieldRecord, MeshRecord, SolveOutcome  # noqa: E402


def _outcome() -> SolveOutcome:
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    topology = np.array([[0, 1, 2]], dtype=np.int64)
    return SolveOutcome(
        observables={"energy": 1.25, "l2_error": 0.5, "h1_seminorm_error": 0.75},
        mesh=MeshRecord("triangle", 2, (1, 1), geometry, topology),
        fields=(
            FieldRecord("u", "H1(order=1)", 1, np.array([0.0, 1.0, 2.0]), geometry),
            FieldRecord("flux", "RT0", 1, np.array([3.0, 4.0]), None),
        ),
        notes={"steps": 8, "iterations": [3, 2], "value": np.float64(1.5)},
    )


def _write(tmp_path):
    result_path = tmp_path / "result.json"
    request = {"schema": "sinbad-oracle-request/1", "case_id": "x", "observables": ["energy"]}
    tool = {"name": "sinbad-oracle-fenicsx", "version": "0.11.0", "normalization_version": 1}
    directory = evidence.write_evidence(
        result_path,
        request=request,
        tool=tool,
        spec=registry.CAPABILITIES["poisson"],
        outcome=_outcome(),
        toolchain={"dolfinx": "0.11.0", "dolfinx_git_commit": "abc"},
        normalization_version=1,
    )
    return result_path, directory


def test_evidence_directory_sits_next_to_the_result_file(tmp_path):
    result_path, directory = _write(tmp_path)
    assert directory == tmp_path / "result.json.evidence"
    assert directory == evidence.evidence_dir_for(result_path)
    assert {p.name for p in directory.iterdir()} == {
        "manifest.json",
        "mesh.npz",
        "solution.npz",
    }


def test_manifest_records_hashes_toolchain_scripts_and_observables(tmp_path):
    result_path, directory = _write(tmp_path)
    manifest = evidence.read_manifest(result_path)
    assert manifest["schema"] == evidence.EVIDENCE_SCHEMA
    assert manifest["capability"] == "poisson"
    assert manifest["sinbad_case"] == "01-poisson"
    assert manifest["toolchain"]["dolfinx_git_commit"] == "abc"
    assert manifest["observables"] == {
        "energy": 1.25,
        "l2_error": 0.5,
        "h1_seminorm_error": 0.75,
    }
    assert manifest["normalization_version"] == 1
    assert set(manifest["normalization"]) == registry.CAPABILITIES["poisson"].observables
    assert manifest["notes"] == {"steps": 8, "iterations": [3, 2], "value": 1.5}

    for key in ("mesh", "solution"):
        path = directory / manifest[key]["file"]
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest[key]["sha256"] == digest
    assert manifest["mesh"]["vertex_count"] == 3 and manifest["mesh"]["cell_count"] == 1

    scripts = manifest["extraction"]["modules"]
    assert set(scripts) == {
        "poisson.py",
        "common.py",
        "evidence.py",
        "adapter.py",
        "protocol.py",
    }
    package = evidence.PACKAGE_DIR
    for name, digest in scripts.items():
        assert digest == "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest()
    assert manifest["extraction"]["package_digest"] == evidence.package_digest(
        manifest["extraction"]["package_scripts"]
    )


def test_solution_arrays_round_trip_with_dof_coordinates_only_for_point_dof_fields(tmp_path):
    _, directory = _write(tmp_path)
    solution = np.load(directory / "solution.npz")
    assert set(solution.files) == {"u", "u__dof_coordinates", "flux"}
    assert solution["u"].tolist() == [0.0, 1.0, 2.0]
    mesh = np.load(directory / "mesh.npz")
    assert mesh["topology"].tolist() == [[0, 1, 2]]
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    fields = {entry["name"]: entry for entry in manifest["solution"]["fields"]}
    assert fields["u"]["dof_coordinates"] == "u__dof_coordinates"
    assert "dof_coordinates" not in fields["flux"]


def test_stale_evidence_is_replaced_wholesale(tmp_path):
    result_path = tmp_path / "result.json"
    stale = evidence.evidence_dir_for(result_path)
    stale.mkdir()
    (stale / "leftover.bin").write_bytes(b"old")
    _write(tmp_path)
    assert not (stale / "leftover.bin").exists()
    assert (stale / "manifest.json").exists()
