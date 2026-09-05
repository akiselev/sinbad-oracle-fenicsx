"""sinbad-oracle-fenicsx executable entrypoint (SV0-C5).

Invoked exactly as `sinbad-oracle-fenicsx <request-file> <result-file>` per
`sinbad-oracle-protocol/2` (see `sinbad/src/oracle.rs`'s `invoke`): bulk
inputs/outputs travel through these two files, never worker JSON frames.

Refusal ordering mirrors the reference fake oracle
(`sinbad/src/bin/sinbad-fake-oracle.rs`) exactly for the identity check, then
extends it with capability/availability/observable/case checks the fake
oracle does not need:

1. tool identity mismatch -> Refused(version_lie). This must come first and
   `result.tool` must always be this adapter's own honestly-reported
   identity: Sinbad's harness treats any tool-identity mismatch between the
   request and the result as a version lie, independent of the `status`
   payload, so identity is checked before anything else.
2. unknown capability (not in `registry.CAPABILITIES`) -> Refused(unsupported_case).
3. dolfinx not importable in this environment -> Refused(unsupported_case).
   The sealed protocol has no distinct "unavailable" refusal class; an
   honest environment-capability refusal is reported through
   `unsupported_case`, never a fabricated result.
4. unsupported observable name(s) for that capability -> Refused(unsupported_case).
5. the capability module itself refuses the case (`outcome.UnsupportedCase`:
   an unaddressable refinement, a declared problem with no solution, a
   nonlinear solve that did not converge) -> Refused(unsupported_case)
   carrying the module's exact reason.
6. otherwise: solve independently with dolfinx, retain the raw evidence
   (`evidence.py`: mesh, solution, extraction-script hashes, full toolchain
   identity) next to the result file, and report Satisfied.

Any other failure (bad arguments, an unreadable/malformed request, an
unexpected exception during solve, a failure to retain evidence) exits with
a non-zero status and no result file, which the harness reads as a `crash`
-- an honest signal that this run did not complete, not a typed refusal.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from . import evidence, protocol, registry
from .capability import probe_dolfinx, toolchain_identity
from .outcome import SolveOutcome, UnsupportedCase

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


class Handled:
    """One handled request: the wire result plus, when satisfied, what to retain."""

    def __init__(
        self,
        result: protocol.OracleResult,
        spec: registry.CapabilitySpec | None = None,
        outcome: SolveOutcome | None = None,
    ) -> None:
        self.result = result
        self.spec = spec
        self.outcome = outcome


def handle_request(request: protocol.OracleRequest) -> Handled:
    actual = _actual_tool_identity()

    def refused(message: str, cls: str = "unsupported_case") -> Handled:
        return Handled(protocol.refused_result(actual, request.case_id, cls, message))

    if request.tool != actual:
        return refused(_identity_mismatch_message(request.tool, actual), "version_lie")

    spec = registry.lookup(request.capability)
    if spec is None:
        return refused(
            f"capability {request.capability!r} is not supported; "
            f"sinbad-oracle-fenicsx implements {sorted(registry.CAPABILITIES)}"
        )

    if actual.version == "unavailable":
        return refused(
            "dolfinx is not importable in this execution environment; cannot "
            f"independently solve the {spec.capability!r} capability here (see INSTALL.md)"
        )

    unsupported = sorted(set(request.observables) - spec.observables)
    if unsupported:
        return refused(
            f"unsupported observable(s) {unsupported} for capability {spec.capability!r}; "
            f"supported: {sorted(spec.observables)}"
        )

    # Deferred: only import dolfinx-backed code once confirmed present.
    module = importlib.import_module(f"{__package__}.{spec.module}")
    try:
        outcome = module.solve(request.refinement)
    except UnsupportedCase as reason:
        return refused(str(reason))

    missing = spec.observables - set(outcome.observables)
    if missing:
        raise RuntimeError(
            f"capability module {spec.module!r} did not produce registered observable(s) "
            f"{sorted(missing)}"
        )
    observables = {name: outcome.observables[name] for name in request.observables}
    return Handled(
        protocol.satisfied_result(actual, request.case_id, observables), spec, outcome
    )


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

    handled = handle_request(request)

    if handled.outcome is not None and handled.spec is not None:
        # Evidence first: a result without its retained evidence would violate the retention
        # contract, so a failure here is a crash (no result file), never a bare number.
        evidence.write_evidence(
            result_path,
            request=request.to_dict(),
            tool=handled.result.tool.to_dict(),
            spec=handled.spec,
            outcome=handled.outcome,
            toolchain=toolchain_identity(),
            normalization_version=NORMALIZATION_VERSION,
        )

    try:
        handled.result.write(result_path)
    except OSError as error:
        print(f"result write failed: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
