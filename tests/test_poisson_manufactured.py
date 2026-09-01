"""dolfinx-dependent correctness tests, skipped when dolfinx is unavailable.

These are the "solve correctness against the closed-form solution" tests:
they check the independently-computed L2 error against the manufactured
solution shrinks under refinement and converges near the expected P1 order,
and they exercise the adapter's full happy path end-to-end. None of this can
run on a host without dolfinx (see the lane report), so every test here
starts with `pytest.importorskip("dolfinx")` and is expected to show as
skipped, not passed, in that environment.
"""

import json
import math
import subprocess
import sys

import pytest

dolfinx = pytest.importorskip("dolfinx")

from sinbad_oracle_fenicsx import protocol  # noqa: E402
from sinbad_oracle_fenicsx.adapter import _actual_tool_identity  # noqa: E402,SLF001
from sinbad_oracle_fenicsx.poisson import solve_manufactured_poisson  # noqa: E402


@pytest.mark.parametrize("refinement", [(4, 4), (8, 8), (16, 16)])
def test_manufactured_solution_energy_is_positive_and_finite(refinement):
    result = solve_manufactured_poisson(refinement)
    assert math.isfinite(result["energy"])
    assert result["energy"] > 0.0
    assert math.isfinite(result["l2_error"])
    assert math.isfinite(result["h1_seminorm_error"])


def test_l2_error_decreases_under_refinement():
    coarse = solve_manufactured_poisson((4, 4))
    fine = solve_manufactured_poisson((16, 16))
    assert fine["l2_error"] < coarse["l2_error"]


def test_l2_error_converges_near_second_order():
    coarse = solve_manufactured_poisson((4, 4))
    fine = solve_manufactured_poisson((8, 8))
    # P1 Lagrange on this manufactured solution: expected L2 order ~2.
    order = math.log(coarse["l2_error"] / fine["l2_error"]) / math.log(2.0)
    assert order > 1.7


def test_adapter_happy_path_end_to_end(tmp_path):
    actual = _actual_tool_identity()
    request = {
        "schema": protocol.ORACLE_REQUEST_SCHEMA,
        "tool": actual.to_dict(),
        "capability": "poisson",
        "case_id": "sv0/d1-poisson",
        "model_digest": "blake3:deadbeef",
        "refinement": [8, 8],
        "observables": ["energy", "l2_error"],
    }
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "sinbad_oracle_fenicsx", str(request_path), str(result_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == {"status": "satisfied"}
    assert set(result["observables"]) == {"energy", "l2_error"}
    energy = protocol.bits_to_finite_f64(result["observables"]["energy"])
    l2_error = protocol.bits_to_finite_f64(result["observables"]["l2_error"])
    assert energy > 0.0
    assert l2_error < 0.1
