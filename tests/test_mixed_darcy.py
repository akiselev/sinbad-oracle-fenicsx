"""dolfinx-dependent tests for the D5 RT0-P0 mixed Darcy capability.

Three things are proved here: the declared 13-mixed-darcy case is refused as
inconsistent (not solved), the same solver converges at first order on a
compatible manufactured problem, and the full-rank p = 0 variant reproduces
the pressure mean Sinbad's own runner records for this case -- the diagnosis
of what Sinbad's compiled `neumann flux = 0` boundary term actually imposes.
"""

import math

import pytest

pytest.importorskip("dolfinx")

import ufl  # noqa: E402

from sinbad_oracle_fenicsx import mixed_darcy  # noqa: E402
from sinbad_oracle_fenicsx.outcome import UnsupportedCase  # noqa: E402

# GX-CONTRACTS C11.21: Sinbad's recorded `pressure_mean` on the same [2,2,2] RT0-P0 mesh.
SINBAD_C11_21_PRESSURE_MEAN = 2.5948e-2


def p_exact(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) * ufl.cos(ufl.pi * x[2])


def u_exact(x):
    return -ufl.grad(p_exact(x))


def f_exact(x):
    return 3.0 * ufl.pi**2 * p_exact(x)


@pytest.fixture(scope="module")
def manufactured():
    return {
        n: mixed_darcy.solve_with(
            (n, n), source_term=f_exact, exact_pressure=p_exact, exact_flux=u_exact
        )
        for n in (2, 4, 8)
    }


def test_declared_case_is_refused_as_inconsistent():
    with pytest.raises(UnsupportedCase) as refusal:
        mixed_darcy.solve((2, 2))
    message = str(refusal.value)
    assert "no solution" in message
    assert "integral(source_term) = 1" in message


def test_manufactured_compatible_problem_converges_at_first_order(manufactured):
    for name in ("pressure_l2_error", "flux_l2_error"):
        order = math.log(manufactured[4].notes[name] / manufactured[8].notes[name]) / math.log(
            2.0
        )
        assert order > 0.85, (name, order)
    assert manufactured[2].notes["system_dimension"] == 168  # 120 RT0 + 48 P0, as Sinbad's


def test_impermeable_walls_carry_no_net_flow_and_pressure_is_gauged(manufactured):
    for outcome in manufactured.values():
        o = outcome.observables
        assert abs(o["total_flow"]) < 1e-12
        assert abs(o["source_compatibility_defect"]) < 1e-8
        assert o["flux_l2_norm"] > 1.0 and o["pressure_l2_norm"] > 0.1


def test_pressure_dirichlet_zero_variant_reproduces_sinbads_c11_21_pressure_mean():
    outcome = mixed_darcy.solve_with(
        (2, 2), source_term=lambda x: 1.0, boundary="pressure_dirichlet_zero"
    )
    assert abs(outcome.notes["pressure_mean_before_gauge"] - SINBAD_C11_21_PRESSURE_MEAN) < 1e-6
    assert (
        abs(outcome.observables["total_flow"] - 1.0) < 1e-12
    )  # the walls leak the unit source
    assert outcome.observables["mass_residual_l2"] < 1e-12
