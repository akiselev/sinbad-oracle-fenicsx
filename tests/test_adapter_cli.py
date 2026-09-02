"""Offline subprocess conformance tests for the adapter entrypoint.

These invoke the real CLI (`python -m sinbad_oracle_fenicsx`) exactly as
Sinbad's harness would (`command request-file result-file`), exercising
every refusal path that is reproducible without dolfinx: identity mismatch,
unsupported capability, environment unavailability, and malformed/missing
request handling. The one path this file cannot exercise is a genuine
`satisfied` solve, which needs dolfinx -- see `test_adapter_live.py` and the
per-capability tests, skipped without dolfinx (run them through
`scripts/dolfinx-image.sh pytest tests`), and `test_recorded_fixtures.py`
for the recorded outputs of those live runs.
"""

import json
import subprocess
import sys

import pytest

from sinbad_oracle_fenicsx import protocol, registry
from sinbad_oracle_fenicsx.adapter import _actual_tool_identity  # noqa: SLF001


def run_adapter(request_path, result_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sinbad_oracle_fenicsx", str(request_path), str(result_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_request(path, **overrides):
    actual = _actual_tool_identity()
    data = {
        "schema": protocol.ORACLE_REQUEST_SCHEMA,
        "tool": actual.to_dict(),
        "capability": "poisson",
        "case_id": "sv0/d1-poisson",
        "model_digest": "blake3:deadbeef",
        "refinement": [4, 4],
        "observables": ["energy"],
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return actual


def test_wrong_argument_count_exits_2(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "sinbad_oracle_fenicsx", str(tmp_path / "only-one-arg")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2


def test_missing_request_file_exits_2_and_writes_no_result(tmp_path):
    request_path = tmp_path / "does-not-exist.json"
    result_path = tmp_path / "result.json"
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 2
    assert not result_path.exists()


def test_malformed_request_json_exits_2_and_writes_no_result(tmp_path):
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text("{not valid json", encoding="utf-8")
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 2
    assert not result_path.exists()


def test_version_lie_is_refused_with_actual_identity_reported(tmp_path):
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    _write_request(
        request_path,
        tool={
            "name": "sinbad-oracle-fenicsx",
            "version": "9.9.9",
            "normalization_version": 999,
        },
    )
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == protocol.ORACLE_RESULT_SCHEMA
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "version_lie"
    # The adapter must report its own honest identity, not the lie.
    assert result["tool"]["version"] != "9.9.9"
    assert result["observables"] == {}


def test_unsupported_capability_is_refused_and_names_the_registry(tmp_path):
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    # "elasticity" is not a registered id (the D3 capability is "linear_elasticity").
    _write_request(request_path, capability="elasticity")
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    assert "'elasticity'" in result["status"]["message"]
    for capability in sorted(registry.CAPABILITIES):
        assert capability in result["status"]["message"]
    assert not (tmp_path / "result.json.evidence").exists()


@pytest.mark.parametrize("capability", sorted(registry.CAPABILITIES))
def test_every_registered_capability_passes_the_capability_gate(tmp_path, capability):
    # With a matching identity every registered capability gets past the capability check;
    # what comes back next depends only on the environment: the honest dolfinx-unavailable
    # refusal on a host without dolfinx, else the unsupported-observable refusal (the probe
    # observable below is defined by no capability), never a crash.
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    actual = _write_request(
        request_path, capability=capability, observables=["not-a-real-observable"]
    )
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    message = result["status"]["message"]
    if actual.version == "unavailable":
        assert "dolfinx" in message and capability in message
    else:
        assert "not-a-real-observable" in message and capability in message
    assert not (tmp_path / "result.json.evidence").exists()


def test_dolfinx_unavailable_or_unsupported_observable_is_refused_not_crashed(tmp_path):
    # On a host without dolfinx (this adapter's own CI), a correctly-matched
    # identity + poisson capability request must still come back as a typed
    # unsupported_case refusal, never a crash and never a fabricated result.
    # On a host with dolfinx, the same request instead exercises the
    # unsupported-observable path via a name no capability will ever define.
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    actual = _write_request(request_path, observables=["not-a-real-observable"])
    completed = run_adapter(request_path, result_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    if actual.version == "unavailable":
        assert "dolfinx" in result["status"]["message"]
    else:
        assert "not-a-real-observable" in result["status"]["message"]
