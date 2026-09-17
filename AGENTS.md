# Agent rules for sinbad-oracle-fenicsx

- This repository owns the Python/FEniCSx oracle adapter. Core numerical runtime
  and ordinary tests must not require Python, network access, or this adapter.
  Optional validation scripts and explicitly enabled live-oracle tests are separate
  from that requirement; unavailable oracle evidence is not a passing result.
- Sinbad owns `sinbad-oracle-protocol/2` (requests `/1` and `/2`, results `/1`) (`sinbad/src/oracle.rs`).
  `protocol.py` mirrors it verbatim; a wire change starts on the Sinbad side.
- One capability per Sinbad case file, registered in `registry.py`, solving
  the *declared* problem from the case's own data with dolfinx. Never read
  Sinbad's numbers, never fabricate a result: a problem the adapter cannot
  honestly solve is a typed `unsupported_case` refusal with the reason.
- Every satisfied run retains raw evidence (`evidence.py`). Observable
  definitions are `NORMALIZATION.md`, generated from the registry
  (`python -m sinbad_oracle_fenicsx.normalization_doc > NORMALIZATION.md`).
- Gate before committing: `ruff check src tests scripts`, `ruff format --check
  src tests scripts`, `PYTHONPATH=src python3 -m pytest tests` (offline), and
  `scripts/dolfinx-image.sh python3 -m pytest tests` when docker is available.
  Re-record `tests/fixtures/recorded` after touching a capability module and
  update `STATUS.md` in the same commit.
- Commit directly on `master`; no feature branches or pull requests.
