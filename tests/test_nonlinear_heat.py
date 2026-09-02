"""dolfinx-dependent correctness tests for the D2 nonlinear-heat capability (skipped otherwise)."""

import math

import pytest

pytest.importorskip("dolfinx")

from sinbad_oracle_fenicsx import nonlinear_heat  # noqa: E402

EXACT_TOTAL_ENERGY = 300.0 + 4.0 / math.pi**2  # integral of 300 + sin(pi x) sin(pi y)


@pytest.fixture(scope="module")
def ladder():
    return {n: nonlinear_heat.solve((n, n)) for n in (2, 4, 8)}


def test_case_authored_forcing_matches_the_symbolic_derivation(ladder):
    for outcome in ladder.values():
        assert outcome.observables["forcing_defect"] < 1e-10


def test_transient_march_stays_on_the_steady_manufactured_state(ladder):
    # Exact steady solution + exact initial condition: the BDF2 trajectory must end on the
    # discrete steady state, so the final-time error equals the steady Newton error closely.
    for outcome in ladder.values():
        o = outcome.observables
        assert abs(o["l2_error"] - o["steady_l2_error"]) < 1e-2 * o["steady_l2_error"]
        assert outcome.notes["steps"] == 8
        assert abs(outcome.notes["final_time"] - 0.4) < 1e-12
        assert all(1 <= it <= 10 for it in outcome.notes["newton_iterations_per_step"])


def test_steady_l2_error_converges_near_second_order(ladder):
    order = math.log(
        ladder[4].observables["steady_l2_error"] / ladder[8].observables["steady_l2_error"]
    ) / math.log(2.0)
    assert order > 1.8


def test_total_energy_converges_to_the_exact_integral(ladder):
    gaps = [abs(ladder[n].observables["total_energy"] - EXACT_TOTAL_ENERGY) for n in (2, 4, 8)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 2e-2


def test_nodal_error_is_a_vertex_rms_bounded_by_the_l2_error(ladder):
    for outcome in ladder.values():
        o = outcome.observables
        assert 0.0 < o["nodal_l2_error"] < o["l2_error"]
