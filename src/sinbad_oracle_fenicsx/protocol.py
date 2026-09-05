"""Typed (de)serialization for the sinbad-oracle-protocol/2 wire contract (requests /1 and /2, results /1).

Sinbad (the frozen product side, see `sinbad/src/oracle.rs`) owns this
schema. This module mirrors it exactly for the FEniCSx adapter and adds no
fields, renames no keys, and reorders nothing relative to the Rust structs.
Field names and enum tag/value strings are copied verbatim from the sealed
protocol; do not "clean them up" independently of a protocol version bump.

Two wire quirks are easy to get wrong translating Rust serde output to
Python and are called out here explicitly:

* `OracleStatus` is a Rust internally-tagged enum
  (`#[serde(tag = "status", rename_all = "snake_case")]`) embedded in
  `OracleResult` under the field name `status`. The result JSON therefore
  nests a `status` object inside a `status` key:
  `{"status": {"status": "satisfied"}}` or
  `{"status": {"status": "refused", "class": ..., "message": ...}}`.
* `FiniteF64` (`sinbad/src/verification_execution.rs`) is a Rust newtype
  tuple struct (`struct FiniteF64(u64)`) with no `#[serde(transparent)]`
  override, so serde's default newtype-struct serialization applies: the
  wire value is the raw IEEE-754 bit pattern of the double as an unsigned
  64-bit JSON integer (`f64::to_bits()`), never a JSON float. Every value in
  `OracleResult.observables` must be encoded/decoded through
  `finite_f64_bits`/`bits_to_finite_f64` below, matching Rust bit-for-bit.
  `FiniteF64::new` also rejects non-finite values and negative zero; this
  module enforces the same rule before encoding.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ORACLE_PROTOCOL_SCHEMA = "sinbad-oracle-protocol/2"
#: `sinbad-oracle-request/2` (Sinbad `17c56c3`, C12.4): `refinement` has two entries for a 2-D
#: case or three for a 3-D case, so a 3-D ladder is addressed exactly. The result document is
#: unchanged at `/1`.
ORACLE_REQUEST_SCHEMA = "sinbad-oracle-request/2"
#: `sinbad-oracle-request/1` (exactly two `refinement` entries) is still answered so the
#: recorded fixtures under `tests/fixtures/recorded/` stay regenerable.
ORACLE_REQUEST_SCHEMA_V1 = "sinbad-oracle-request/1"
ACCEPTED_REQUEST_SCHEMAS = frozenset({ORACLE_REQUEST_SCHEMA_V1, ORACLE_REQUEST_SCHEMA})
ORACLE_RESULT_SCHEMA = "sinbad-oracle-result/1"

# The protocol's `OracleRefusalClass` enum, `snake_case`-rendered.
REFUSAL_CLASSES = frozenset(
    {"version_lie", "crash", "timeout", "malformed_output", "unsupported_case"}
)


class ProtocolError(ValueError):
    """A request/result document does not match sinbad-oracle-protocol/2 (or the /1 request)."""


def finite_f64_bits(value: float) -> int:
    """Encodes `value` as the u64 bit pattern `FiniteF64::new` would store.

    Mirrors `FiniteF64::new` in `sinbad/src/verification_execution.rs`
    exactly: non-finite values and negative zero are rejected.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(f"observable value must be a real number, got {value!r}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ProtocolError(f"observable value must be finite, got {value!r}")
    if value == 0.0 and struct.pack(">d", value) == struct.pack(">d", -0.0):
        raise ProtocolError("observable value must not be negative zero")
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def bits_to_finite_f64(bits: int) -> float:
    """Decodes a `FiniteF64` wire integer back to a Python float."""
    if not isinstance(bits, int) or isinstance(bits, bool) or not (0 <= bits < 2**64):
        raise ProtocolError(f"FiniteF64 wire value must be a u64, got {bits!r}")
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


@dataclass(frozen=True)
class OracleToolIdentity:
    name: str
    version: str
    normalization_version: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "normalization_version": self.normalization_version,
        }

    @staticmethod
    def from_dict(data: Mapping) -> "OracleToolIdentity":
        try:
            return OracleToolIdentity(
                name=str(data["name"]),
                version=str(data["version"]),
                normalization_version=int(data["normalization_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"malformed tool identity: {error}") from error


@dataclass(frozen=True)
class OracleRequest:
    schema: str
    tool: OracleToolIdentity
    capability: str
    case_id: str
    model_digest: str
    refinement: tuple[int, ...]
    observables: tuple[str, ...]

    @staticmethod
    def from_dict(data: Mapping) -> "OracleRequest":
        try:
            schema = str(data["schema"])
            if schema not in ACCEPTED_REQUEST_SCHEMAS:
                raise ProtocolError(
                    f"unsupported request schema {schema!r}, expected one of "
                    f"{sorted(ACCEPTED_REQUEST_SCHEMAS)}"
                )
            tool = OracleToolIdentity.from_dict(data["tool"])
            capability = str(data["capability"])
            case_id = str(data["case_id"])
            model_digest = str(data["model_digest"])
            refinement_raw = list(data["refinement"])
            if schema == ORACLE_REQUEST_SCHEMA_V1:
                if len(refinement_raw) != 2:
                    raise ProtocolError(
                        f"{ORACLE_REQUEST_SCHEMA_V1} refinement must have exactly two entries"
                    )
            elif len(refinement_raw) not in (2, 3):
                raise ProtocolError(
                    f"{ORACLE_REQUEST_SCHEMA} refinement must have two (2-D) or three (3-D) entries"
                )
            refinement = tuple(int(entry) for entry in refinement_raw)
            observables = tuple(str(name) for name in data["observables"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"malformed oracle request: {error}") from error
        return OracleRequest(
            schema=schema,
            tool=tool,
            capability=capability,
            case_id=case_id,
            model_digest=model_digest,
            refinement=refinement,
            observables=observables,
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "tool": self.tool.to_dict(),
            "capability": self.capability,
            "case_id": self.case_id,
            "model_digest": self.model_digest,
            "refinement": list(self.refinement),
            "observables": list(self.observables),
        }

    @staticmethod
    def read(path: Path) -> "OracleRequest":
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ProtocolError(f"could not read request file {path}: {error}") from error
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"request file {path} is not valid JSON: {error}") from error
        if not isinstance(data, Mapping):
            raise ProtocolError(f"request file {path} must decode to a JSON object")
        return OracleRequest.from_dict(data)


@dataclass(frozen=True)
class OracleRefusal:
    cls: str
    message: str

    def __post_init__(self) -> None:
        if self.cls not in REFUSAL_CLASSES:
            raise ProtocolError(f"unknown refusal class {self.cls!r}")


@dataclass(frozen=True)
class OracleResult:
    schema: str
    tool: OracleToolIdentity
    case_id: str
    satisfied: bool
    refusal: OracleRefusal | None
    observables: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.satisfied and self.refusal is not None:
            raise ProtocolError("a satisfied result cannot carry a refusal")
        if not self.satisfied and self.refusal is None:
            raise ProtocolError("a refused result must carry a refusal")

    def to_dict(self) -> dict:
        if self.satisfied:
            status: dict = {"status": "satisfied"}
        else:
            assert self.refusal is not None
            status = {
                "status": "refused",
                "class": self.refusal.cls,
                "message": self.refusal.message,
            }
        return {
            "schema": self.schema,
            "tool": self.tool.to_dict(),
            "case_id": self.case_id,
            "status": status,
            "observables": {
                name: finite_f64_bits(value) for name, value in self.observables.items()
            },
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def satisfied_result(
    tool: OracleToolIdentity, case_id: str, observables: Mapping[str, float]
) -> OracleResult:
    return OracleResult(
        schema=ORACLE_RESULT_SCHEMA,
        tool=tool,
        case_id=case_id,
        satisfied=True,
        refusal=None,
        observables=dict(observables),
    )


def refused_result(
    tool: OracleToolIdentity, case_id: str, cls: str, message: str
) -> OracleResult:
    return OracleResult(
        schema=ORACLE_RESULT_SCHEMA,
        tool=tool,
        case_id=case_id,
        satisfied=False,
        refusal=OracleRefusal(cls=cls, message=message),
        observables={},
    )
