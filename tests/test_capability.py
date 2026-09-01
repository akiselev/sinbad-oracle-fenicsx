"""Offline tests for the dolfinx capability probe."""

import sys

from sinbad_oracle_fenicsx.capability import probe_dolfinx


def test_probe_reports_unavailable_when_dolfinx_cannot_be_imported(monkeypatch):
    # `sys.modules[name] = None` makes a subsequent `import name` raise
    # ImportError immediately, per the Python import system's own contract,
    # without needing dolfinx to be absent from the real environment.
    monkeypatch.setitem(sys.modules, "dolfinx", None)
    result = probe_dolfinx()
    assert result.available is False
    assert result.version is None
    assert result.reason


def test_probe_never_raises_in_this_environment():
    # This documents the current host's real capability honestly rather than
    # asserting a specific outcome nobody has verified either way here; see
    # the lane report for what this adapter's own CI host actually has.
    result = probe_dolfinx()
    assert isinstance(result.available, bool)
    if not result.available:
        assert result.version is None
        assert result.reason
