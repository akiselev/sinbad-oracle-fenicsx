"""sinbad-oracle-fenicsx executable entrypoint (SV0-C5).

Invoked exactly as `sinbad-oracle-fenicsx <request-file> <result-file>` per
`sinbad-oracle-protocol/1` (see `sinbad/src/oracle.rs`'s `invoke`): bulk
inputs/outputs travel through these two files, never worker JSON frames.

Refusal ordering mirrors the reference fake oracle
(`sinbad/src/bin/sinbad-fake-oracle.rs`) exactly for the identity check, then
extends it with capability/availability/observable checks the fake oracle
does not need:

1. tool identity mismatch -> Refused(version_lie). This must come first and
   `result.tool` must always be this adapter's own honestly-reported
   identity: Sinbad's harness treats any tool-identity mismatch between the
   request and the result as a version lie, independent of the `status`
   payload, so identity is checked before anything else.
2. unsupported capability -> Refused(unsupported_case).
3. dolfinx not importable in this environment -> Refused(unsupported_case).
   The sealed protocol has no distinct "unavailable" refusal class; an
   honest environment-capability refusal is reported through
   `unsupported_case`, never a fabricated result.
4. unsupported observable name(s) requested -> Refused(unsupported_case).
5. otherwise: solve independently with dolfinx and report Satisfied.

Any other failure (bad arguments, an unreadable/malformed request, an
unexpected exception during solve) exits with a non-zero status and no
result file, which the harness reads as a `crash` -- an honest signal that
this run did not complete, not a typed refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import protocol
from .capability import probe_dolfinx

ADAPTER_NAME = "sinbad-oracle-fenicsx"
NORMALIZATION_VERSION = 1


def _actual_tool_identity() -> protocol.OracleToolIdentity:
    availability = probe_dolfinx()
    version = availability.version if availability.available else "unavailable"
    return protocol.OracleToolIdentity(
        name=ADAPTER_NAME,
        version=version,
        normalization_version=NORMALIZATION_VERSION,
    )


def _identity_mismatch_message(
    requested: protocol.OracleToolIdentity, actual: protocol.OracleToolIdentity
) -> str:
    return (
        f"requested {requested.name}@{requested.version} "
        f"(normalization {requested.normalization_version}), running "
        f"{actual.name}@{actual.version} (normalization {actual.normalization_version})"
    )


def handle_request(request: protocol.OracleRequest) -> protocol.OracleResult:
    actual = _actual_tool_identity()

    if request.tool != actual:
        return protocol.refused_result(
            actual,
            request.case_id,
            "version_lie",
            _identity_mismatch_message(request.tool, actual),
        )

    if request.capability != protocol.POISSON_CAPABILITY:
        return protocol.refused_result(
            actual,
            request.case_id,
            "unsupported_case",
            f"capability {request.capability!r} is not supported; "
            f"sinbad-oracle-fenicsx implements {protocol.POISSON_CAPABILITY!r} only",
        )

    if actual.version == "unavailable":
        return protocol.refused_result(
            actual,
            request.case_id,
            "unsupported_case",
            "dolfinx is not importable in this execution environment; "
            "cannot independently solve the poisson capability here "
            "(see INSTALL.md)",
        )

    from . import poisson  # deferred: only import dolfinx once confirmed present

    unsupported = sorted(set(request.observables) - poisson.SUPPORTED_OBSERVABLES)
    if unsupported:
        return protocol.refused_result(
            actual,
            request.case_id,
            "unsupported_case",
            f"unsupported observable(s) {unsupported}; supported: "
            f"{sorted(poisson.SUPPORTED_OBSERVABLES)}",
        )

    solved = poisson.solve_manufactured_poisson(request.refinement)
    observables = {name: solved[name] for name in request.observables}
    return protocol.satisfied_result(actual, request.case_id, observables)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) != 3:
        print("usage: sinbad-oracle-fenicsx request-file result-file", file=sys.stderr)
        return 2

    request_path = Path(argv[1])
    result_path = Path(argv[2])

    try:
        request = protocol.OracleRequest.read(request_path)
    except protocol.ProtocolError as error:
        print(f"request read failed: {error}", file=sys.stderr)
        return 2

    result = handle_request(request)

    try:
        result.write(result_path)
    except OSError as error:
        print(f"result write failed: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
