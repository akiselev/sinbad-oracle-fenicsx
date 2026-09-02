"""dolfinx-dependent correctness tests for the D3 linear-elasticity capability."""

import math

import pytest

pytest.importorskip("dolfinx")

from sinbad_oracle_fenicsx import linear_elasticity  # noqa: E402
from sinbad_oracle_fenicsx.outcome import UnsupportedCase  # noqa: E402


@pytest.fixture(scope="module")
def ladder():
    return {n: linear_elasticity.solve((n, n)) for n in (2, 4, 8)}


def test_case_authored_body_force_matches_the_symbolic_derivation(ladder):
    for outcome in ladder.values():
        assert outcome.observables["forcing_defect"] < 1e-10


def test_isotropic_refinement_reads_n_n_as_n_n_n(ladder):
    assert ladder[2].mesh.subdivisions == (2, 2, 2)
    assert ladder[2].mesh.cell_type == "tetrahedron"
    assert ladder[2].mesh.topology.shape == (2 * 2 * 2 * 6, 4)


def test_anisotropic_refinement_is_refused_not_guessed():
    with pytest.raises(UnsupportedCase):
        linear_elasticity.solve((2, 4))


def test_l2_error_converges_near_second_order(ladder):
    order = math.log(
        ladder[4].observables["l2_error"] / ladder[8].observables["l2_error"]
    ) / math.log(2.0)
    assert order > 1.7


def test_h1_seminorm_error_converges_near_first_order(ladder):
    order = math.log(
        ladder[4].observables["h1_seminorm_error"] / ladder[8].observables["h1_seminorm_error"]
    ) / math.log(2.0)
    assert order > 0.8


def test_energy_observables_are_positive_and_increase_toward_the_continuum(ladder):
    # The exact integral of dot(u, u) is 3/8; P1 underestimates it on coarse meshes.
    for n in (2, 4, 8):
        o = ladder[n].observables
        assert o["strain_energy"] > 0.0 and o["displacement_magnitude_sq"] > 0.0
    assert (
        ladder[2].observables["displacement_magnitude_sq"]
        < ladder[4].observables["displacement_magnitude_sq"]
        < ladder[8].observables["displacement_magnitude_sq"]
        < 3.0 / 8.0
    )
