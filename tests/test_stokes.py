"""dolfinx-dependent correctness tests for the D4 Taylor-Hood Stokes capability."""

import pytest

pytest.importorskip("dolfinx")

from sinbad_oracle_fenicsx import stokes  # noqa: E402

# Sinbad's own recorded 59-dof solution RMS on the identical [2, 2] Taylor-Hood mesh
# (GX-CONTRACTS C11.20, "3.2595e-2", zero-arithmetic-mean pressure gauge).
SINBAD_C11_20_SOLUTION_RMS = 3.2595e-2


@pytest.fixture(scope="module")
def ladder():
    return {n: stokes.solve((n, n)) for n in (2, 4, 8)}


def test_taylor_hood_dof_count_matches_sinbads_realization(ladder):
    assert ladder[2].notes["system_dimension"] == 59  # 25 P2 nodes x 2 + 9 P1 nodes


def test_enclosed_cavity_conserves_mass_exactly_and_divergence_shrinks(ladder):
    for outcome in ladder.values():
        assert abs(outcome.observables["mass_defect"]) < 1e-14
    assert (
        ladder[2].observables["divergence_l2_norm"]
        > ladder[4].observables["divergence_l2_norm"]
        > ladder[8].observables["divergence_l2_norm"]
    )


def test_curl_body_force_drives_genuine_flow(ladder):
    for outcome in ladder.values():
        assert outcome.observables["dissipation"] > 1e-5
        assert outcome.observables["velocity_l2_norm"] > 1e-4


def test_pressure_is_gauged_to_zero_mean(ladder):
    import ufl
    from dolfinx import fem

    from sinbad_oracle_fenicsx import common

    for outcome in ladder.values():
        pressure = next(f for f in outcome.fields if f.name == "pressure")
        assert pressure.values.shape[0] == (outcome.mesh.subdivisions[0] + 1) ** 2
    # Recompute the gauge on a fresh solve rather than trusting the stored array.
    domain = common.unit_box((4, 4))
    space = fem.functionspace(domain, ("Lagrange", 1))
    p = fem.Function(space)
    p.x.array[:] = next(f for f in ladder[4].fields if f.name == "pressure").values
    assert abs(common.integral(domain, p * ufl.dx)) < 1e-12


def test_solution_rms_reproduces_sinbads_c11_20_record(ladder):
    rms = ladder[2].observables["solution_rms"]
    assert abs(rms - SINBAD_C11_20_SOLUTION_RMS) < 1e-6, rms
