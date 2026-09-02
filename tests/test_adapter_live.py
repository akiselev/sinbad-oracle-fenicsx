"""dolfinx-dependent end-to-end runs of the real CLI for every capability, with evidence."""

import hashlib
import json
import subprocess
import sys

import pytest

pytest.importorskip("dolfinx")

from sinbad_oracle_fenicsx import evidence, protocol, registry  # noqa: E402
from sinbad_oracle_fenicsx.adapter import _actual_tool_identity  # noqa: E402,SLF001


def run_adapter(tmp_path, capability, refinement, observables, **overrides):
    actual = _actual_tool_identity()
    request = {
        "schema": protocol.ORACLE_REQUEST_SCHEMA,
        "tool": actual.to_dict(),
        "capability": capability,
        "case_id": f"{registry.CAPABILITIES[capability].sinbad_case}/live-test",
        "model_digest": "blake3:live-test",
        "refinement": list(refinement),
        "observables": list(observables),
    }
    request.update(overrides)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "sinbad_oracle_fenicsx", str(request_path), str(result_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8")), result_path


@pytest.mark.parametrize(
    "capability", ["poisson", "nonlinear_heat", "linear_elasticity", "stokes"]
)
def test_satisfied_run_reports_requested_observables_and_retains_evidence(tmp_path, capability):
    spec = registry.CAPABILITIES[capability]
    requested = sorted(spec.observables)[:2]
    result, result_path = run_adapter(tmp_path, capability, (2, 2), requested)
    assert result["status"] == {"status": "satisfied"}
    assert set(result["observables"]) == set(requested)

    manifest = evidence.read_manifest(result_path)
    directory = evidence.evidence_dir_for(result_path)
    for key in ("mesh", "solution"):
        digest = (
            "sha256:"
            + hashlib.sha256((directory / manifest[key]["file"]).read_bytes()).hexdigest()
        )
        assert manifest[key]["sha256"] == digest
    assert manifest["toolchain"]["dolfinx"] == result["tool"]["version"]
    assert manifest["request"]["observables"] == requested
    # The manifest keeps every registered observable even when fewer were requested.
    assert set(manifest["observables"]) == spec.observables
    for name in requested:
        assert manifest["observables"][name] == protocol.bits_to_finite_f64(
            result["observables"][name]
        )


def test_darcy_case_is_refused_with_no_evidence_directory(tmp_path):
    result, result_path = run_adapter(tmp_path, "mixed_darcy", (2, 2), ["flux_l2_norm"])
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    assert "no solution" in result["status"]["message"]
    assert not evidence.evidence_dir_for(result_path).exists()


def test_anisotropic_3d_refinement_is_refused_typed(tmp_path):
    result, _ = run_adapter(tmp_path, "linear_elasticity", (2, 3), ["l2_error"])
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    assert "[2, 3]" in result["status"]["message"]


def test_identity_probe_refuses_version_lie_with_real_identity(tmp_path):
    result, _ = run_adapter(
        tmp_path,
        "stokes",
        (2, 2),
        [],
        tool={"name": "sinbad-oracle-fenicsx", "version": "probe", "normalization_version": 1},
    )
    assert result["status"]["class"] == "version_lie"
    assert result["tool"] == _actual_tool_identity().to_dict()
