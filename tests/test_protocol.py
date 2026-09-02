"""Offline tests for the sinbad-oracle-protocol/1 wire (de)serialization.

None of these tests import dolfinx; they exercise exactly what ordinary,
license-free, network-free CI can run.
"""

import json
import math

import pytest

from sinbad_oracle_fenicsx import protocol

# --- FiniteF64 bit-pattern encoding -----------------------------------------


def test_finite_f64_round_trips_bit_for_bit():
    for value in [0.0, 1.0, -1.0, 1.004, math.pi, 1e-300, 1e300, -123.456]:
        bits = protocol.finite_f64_bits(value)
        assert isinstance(bits, int)
        assert 0 <= bits < 2**64
        assert protocol.bits_to_finite_f64(bits) == value


def test_finite_f64_matches_known_ieee754_bit_patterns():
    # Cross-checked against the IEEE-754 binary64 standard independently of
    # this module's own round-trip, since `f64::to_bits()` on the Rust side
    # is exactly the standard bit pattern, not an implementation detail.
    assert protocol.finite_f64_bits(1.0) == 0x3FF0000000000000
    assert protocol.finite_f64_bits(0.0) == 0x0000000000000000
    assert protocol.finite_f64_bits(-1.0) == 0xBFF0000000000000
    assert protocol.bits_to_finite_f64(0x4000000000000000) == 2.0


def test_finite_f64_rejects_nan_and_infinity():
    with pytest.raises(protocol.ProtocolError):
        protocol.finite_f64_bits(float("nan"))
    with pytest.raises(protocol.ProtocolError):
        protocol.finite_f64_bits(float("inf"))
    with pytest.raises(protocol.ProtocolError):
        protocol.finite_f64_bits(float("-inf"))


def test_finite_f64_rejects_negative_zero():
    with pytest.raises(protocol.ProtocolError):
        protocol.finite_f64_bits(-0.0)
    # Positive zero remains fine.
    assert protocol.finite_f64_bits(0.0) == 0


def test_bits_to_finite_f64_rejects_out_of_range():
    with pytest.raises(protocol.ProtocolError):
        protocol.bits_to_finite_f64(-1)
    with pytest.raises(protocol.ProtocolError):
        protocol.bits_to_finite_f64(2**64)


# --- OracleToolIdentity ------------------------------------------------------


def test_tool_identity_round_trip():
    tool = protocol.OracleToolIdentity(
        name="sinbad-oracle-fenicsx", version="0.7.3", normalization_version=1
    )
    decoded = protocol.OracleToolIdentity.from_dict(tool.to_dict())
    assert decoded == tool


def test_tool_identity_rejects_missing_field():
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleToolIdentity.from_dict({"name": "x", "version": "1"})


# --- OracleRequest ------------------------------------------------------------


def _valid_request_dict(**overrides):
    data = {
        "schema": protocol.ORACLE_REQUEST_SCHEMA,
        "tool": {
            "name": "sinbad-oracle-fenicsx",
            "version": "0.7.3",
            "normalization_version": 1,
        },
        "capability": "poisson",
        "case_id": "sv0/d1-poisson",
        "model_digest": "blake3:deadbeef",
        "refinement": [8, 8],
        "observables": ["energy"],
    }
    data.update(overrides)
    return data


def test_request_from_dict_happy_path():
    request = protocol.OracleRequest.from_dict(_valid_request_dict())
    assert request.schema == protocol.ORACLE_REQUEST_SCHEMA
    assert request.tool.name == "sinbad-oracle-fenicsx"
    assert request.capability == "poisson"
    assert request.refinement == (8, 8)
    assert request.observables == ("energy",)


def test_request_to_dict_round_trips():
    data = _valid_request_dict(observables=["energy", "l2_error"])
    request = protocol.OracleRequest.from_dict(data)
    assert request.to_dict() == data
    assert protocol.OracleRequest.from_dict(request.to_dict()) == request


def test_request_from_dict_rejects_wrong_schema():
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleRequest.from_dict(_valid_request_dict(schema="other/1"))


def test_request_from_dict_rejects_bad_refinement_length():
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleRequest.from_dict(_valid_request_dict(refinement=[8]))


def test_request_from_dict_rejects_missing_field():
    data = _valid_request_dict()
    del data["case_id"]
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleRequest.from_dict(data)


def test_request_read_rejects_missing_file(tmp_path):
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleRequest.read(tmp_path / "does-not-exist.json")


def test_request_read_rejects_invalid_json(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(protocol.ProtocolError):
        protocol.OracleRequest.read(path)


def test_request_read_round_trips_via_file(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_valid_request_dict()), encoding="utf-8")
    request = protocol.OracleRequest.read(path)
    assert request.case_id == "sv0/d1-poisson"


# --- OracleResult -------------------------------------------------------------


def _tool() -> protocol.OracleToolIdentity:
    return protocol.OracleToolIdentity(
        name="sinbad-oracle-fenicsx", version="0.7.3", normalization_version=1
    )


def test_satisfied_result_wire_shape():
    result = protocol.satisfied_result(_tool(), "sv0/d1-poisson", {"energy": 1.5})
    encoded = result.to_dict()
    assert encoded["schema"] == protocol.ORACLE_RESULT_SCHEMA
    assert encoded["status"] == {"status": "satisfied"}
    assert encoded["observables"] == {"energy": protocol.finite_f64_bits(1.5)}


def test_refused_result_wire_shape():
    result = protocol.refused_result(
        _tool(), "sv0/d1-poisson", "unsupported_case", "dolfinx unavailable"
    )
    encoded = result.to_dict()
    assert encoded["status"] == {
        "status": "refused",
        "class": "unsupported_case",
        "message": "dolfinx unavailable",
    }
    assert encoded["observables"] == {}


def test_refused_result_rejects_unknown_class():
    with pytest.raises(protocol.ProtocolError):
        protocol.refused_result(_tool(), "case", "not_a_real_class", "message")


def test_result_write_produces_valid_json(tmp_path):
    result = protocol.satisfied_result(_tool(), "sv0/d1-poisson", {"energy": 2.0})
    path = tmp_path / "result.json"
    result.write(path)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["case_id"] == "sv0/d1-poisson"
    assert decoded["observables"]["energy"] == protocol.finite_f64_bits(2.0)
