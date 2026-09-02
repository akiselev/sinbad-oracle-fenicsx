# Agent rules for sinbad-oracle-fenicsx

- This repository is the only place Python lives in the Sinbad federation.
  Core Sinbad CI must stay Python- and network-free; never make a Sinbad-side
  test depend on this repository being installed.
- Sinbad owns `sinbad-oracle-protocol/1` (`sinbad/src/oracle.rs`).
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
