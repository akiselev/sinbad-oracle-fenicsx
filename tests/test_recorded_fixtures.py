"""Offline protocol tests over recorded live runs (`tests/fixtures/recorded/`).

Each fixture directory holds the exact request the adapter was given, the
exact result file it wrote, and the evidence manifest it retained, recorded
by `scripts/record_live_runs.py` inside the official dolfinx image. These
tests never re-run dolfinx: they prove the recorded wire documents decode
per `sinbad-oracle-protocol/1`, that manifest and result agree bit for bit,
and that the recorded numbers match the values STATUS.md reports (so a
silent re-record that changes a number fails here).
"""

import json
import math
from pathlib import Path

import pytest

from sinbad_oracle_fenicsx import evidence, protocol, registry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "recorded"
RECORDED_IDENTITY = {
    "name": "sinbad-oracle-fenicsx",
    "version": "0.11.0.post0",
    "normalization_version": 1,
}

# Values Sinbad's own contracts record for the same cases; see STATUS.md for provenance.
SINBAD_RECORDED = {
    # GX-CONTRACTS C11.18: FEniCSx `energy` at the [4,4]/[8,8]/[16,16] ladder.
    ("poisson", "4x4", "energy"): (2.1158194763, 1e-9),
    ("poisson", "8x8", "energy"): (2.3741762218, 1e-9),
    ("poisson", "16x16", "energy"): (2.4437400715, 1e-9),
    # GX-CONTRACTS C11.20: Sinbad's own 59-dof Stokes solution RMS at [2,2].
    ("stokes", "2x2", "solution_rms"): (3.2595e-2, 1e-6),
}


def fixture_dirs():
    return sorted(p for p in FIXTURES.glob("*/*") if (p / "result.json").is_file())


@pytest.mark.parametrize("fixture", fixture_dirs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_recorded_result_decodes_per_protocol(fixture):
    request = protocol.OracleRequest.from_dict(
        json.loads((fixture / "request.json").read_text())
    )
    result = json.loads((fixture / "result.json").read_text())
    assert result["schema"] == protocol.ORACLE_RESULT_SCHEMA
    assert result["tool"] == RECORDED_IDENTITY
    assert result["tool"] == request.tool.to_dict()
    assert result["case_id"] == request.case_id
    assert request.capability == fixture.parent.name
    assert request.capability in registry.CAPABILITIES
    status = result["status"]
    if status["status"] == "satisfied":
        assert set(result["observables"]) == set(request.observables)
        for name, bits in result["observables"].items():
            value = protocol.bits_to_finite_f64(bits)
            assert math.isfinite(value), name
    else:
        assert status["status"] == "refused"
        assert status["class"] in protocol.REFUSAL_CLASSES
        assert result["observables"] == {}


@pytest.mark.parametrize("fixture", fixture_dirs(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_recorded_manifest_agrees_with_the_result(fixture):
    manifest_path = fixture / "manifest.json"
    result = json.loads((fixture / "result.json").read_text())
    if result["status"]["status"] != "satisfied":
        assert not manifest_path.exists(), "a refusal retains no solution evidence"
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == evidence.EVIDENCE_SCHEMA
    assert manifest["tool"] == result["tool"]
    assert manifest["request"]["case_id"] == result["case_id"]
    assert manifest["toolchain"]["dolfinx"] == RECORDED_IDENTITY["version"]
    assert manifest["toolchain"]["dolfinx_git_commit"]
    assert manifest["normalization_version"] == RECORDED_IDENTITY["normalization_version"]
    spec = registry.CAPABILITIES[manifest["capability"]]
    assert set(manifest["observables"]) == spec.observables
    assert set(manifest["normalization"]) == spec.observables
    for name, bits in result["observables"].items():
        assert manifest["observables"][name] == protocol.bits_to_finite_f64(bits)
    assert manifest["mesh"]["sha256"].startswith("sha256:")
    assert manifest["solution"]["sha256"].startswith("sha256:")
    assert manifest["extraction"]["modules"][f"{spec.module}.py"].startswith("sha256:")
    assert manifest["mesh"]["subdivisions"] == (
        list(manifest["request"]["refinement"])
        if spec.dimension == 2
        else [manifest["request"]["refinement"][0]] * 3
    )


def test_recorded_values_match_sinbads_own_contract_records():
    for (capability, level, name), (expected, tolerance) in SINBAD_RECORDED.items():
        result = json.loads((FIXTURES / capability / level / "result.json").read_text())
        value = protocol.bits_to_finite_f64(result["observables"][name])
        assert abs(value - expected) <= tolerance, (capability, level, name, value, expected)


def test_recorded_forcing_defects_vanish():
    # The hand-derived case forcings (`03-nonlinear-heat.toml` `source/Q`,
    # `17-linear-elasticity.toml` `source/body_force`) agree with UFL's symbolic derivation.
    for capability in ("nonlinear_heat", "linear_elasticity"):
        for level in (FIXTURES / capability).iterdir():
            result = json.loads((level / "result.json").read_text())
            defect = protocol.bits_to_finite_f64(result["observables"]["forcing_defect"])
            assert defect < 1e-10, (capability, level.name, defect)


def test_recorded_darcy_case_is_refused_as_inconsistent():
    result = json.loads((FIXTURES / "mixed_darcy" / "2x2" / "result.json").read_text())
    assert result["status"]["status"] == "refused"
    assert result["status"]["class"] == "unsupported_case"
    assert "no solution" in result["status"]["message"]
    assert "integral(source_term) = 1" in result["status"]["message"]
